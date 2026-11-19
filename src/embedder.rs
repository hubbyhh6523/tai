use crate::health::Health;
use crate::session_pool::SessionPool;
use anyhow::{Context, Result, anyhow};
use ort::environment::Environment;
use ort::value::Tensor;
use std::path::Path;
use std::time::Duration;
use tokenizers::{Encoding, PaddingParams, PaddingStrategy, Tokenizer, TruncationParams};

pub struct Embedder {
    _env: Environment,
    pool: SessionPool,
    tokenizer: Tokenizer,
    pub dim: usize,
    max_tokens_per_run: usize,
    has_token_type: bool,
    output_name: String,
    pub health: Health,
}

pub struct Embedded {
    pub vectors: Vec<Vec<f32>>,
    /// Non-padding token count per input text.
    pub tokens: Vec<usize>,
}

/// What the HTTP layer needs from the model, so handlers can be tested without loading one.
///
/// Texts are borrowed (`&str`) so the HTTP layer can pass its deduplicated
/// slice straight through without cloning every string a second time.
pub trait Embed: Send + Sync {
    fn embed(&self, texts: &[&str]) -> Result<Embedded>;
    fn health_check(&self) -> Result<(), String>;
    fn consecutive_failures(&self) -> u64;
}

impl Embed for Embedder {
    fn embed(&self, texts: &[&str]) -> Result<Embedded> {
        Embedder::embed(self, texts)
    }

    fn health_check(&self) -> Result<(), String> {
        Embedder::health_check(self)
    }

    fn consecutive_failures(&self) -> u64 {
        self.health.consecutive_failures()
    }
}

fn oe<E: std::fmt::Display>(e: E) -> anyhow::Error {
    anyhow!("{e}")
}

/// Split texts into runs whose padded size stays within `budget`.
///
/// ONNX runs on a dense `[batch, max_len]` tensor, so the cost of a run
/// is `batch * max_len` (padded positions), not the sum of token counts:
/// one 512-token text in a 256-text batch pads the whole batch to 512
/// (131k positions). Indices are sorted longest-first so similar lengths
/// land in the same run; a run is closed before adding a text that would
/// push `batch * longest_in_run` over the budget. Every index appears
/// exactly once. A single text longer than the budget still gets its own
/// run rather than being dropped.
pub fn chunk_by_token_budget(token_lens: &[usize], budget: usize) -> Vec<Vec<usize>> {
    let budget = budget.max(1);
    let mut order: Vec<usize> = (0..token_lens.len()).collect();
    // Longest first; break ties by original index so the order is stable.
    order.sort_by(|&a, &b| token_lens[b].cmp(&token_lens[a]).then(a.cmp(&b)));
    let mut chunks: Vec<Vec<usize>> = Vec::new();
    let mut current: Vec<usize> = Vec::new();
    for idx in order {
        if let Some(&first) = current.first() {
            // Adding idx would pad the whole run to token_lens[first].
            let padded = (current.len() + 1).saturating_mul(token_lens[first]);
            if padded > budget {
                chunks.push(std::mem::take(&mut current));
            }
        }
        current.push(idx);
    }
    if !current.is_empty() {
        chunks.push(current);
    }
    chunks
}

impl Embedder {
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        model_dir: &Path,
        pool_size: usize,
        threads_per_session: usize,
        queue_timeout: Option<Duration>,
        dim: usize,
        max_len: usize,
        max_tokens_per_run: usize,
        max_failures: u64,
    ) -> Result<Self> {
        let mut tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json"))
            .map_err(|e| anyhow!("load tokenizer: {e}"))?;
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: max_len,
                ..Default::default()
            }))
            .map_err(|e| anyhow!("truncation: {e}"))?;
        tokenizer.with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::BatchLongest,
            pad_id: 0,
            pad_token: "[PAD]".into(),
            ..Default::default()
        }));

        let env = ort::init().with_name("embed-svc").build().map_err(oe)?;
        let pool = SessionPool::new(
            &env,
            &model_dir.join("model.onnx"),
            pool_size,
            threads_per_session,
            queue_timeout,
        )
        .map_err(oe)
        .context("load model.onnx")?;

        let probe = pool.acquire().map_err(oe)?;
        let session = probe.session().map_err(oe)?;
        let has_token_type = session
            .inputs()
            .iter()
            .any(|i| i.name() == "token_type_ids");
        let output_name = session
            .outputs()
            .first()
            .map(|o| o.name().to_string())
            .ok_or_else(|| anyhow!("model has no outputs"))?;
        drop(session);
        drop(probe);

        Ok(Self {
            _env: env,
            pool,
            tokenizer,
            dim,
            max_tokens_per_run: max_tokens_per_run.max(1),
            has_token_type,
            output_name,
            health: Health::new(max_failures),
        })
    }

    /// Runs two real inferences before the service starts listening: a short text (checks the output
    /// dimension and normalization) and a text long enough to reach `max_tokens` (checks the model really
    /// supports that sequence length).
    pub fn self_check(&self, max_tokens: usize) -> Result<()> {
        let short = self.embed(&["自检"]).context("short-text inference")?;
        let v = short
            .vectors
            .first()
            .ok_or_else(|| anyhow!("no vector returned"))?;
        if v.len() != self.dim || v.iter().any(|x| !x.is_finite()) {
            return Err(anyhow!(
                "bad vector: len {} (DIM={}), all finite: {}",
                v.len(),
                self.dim,
                v.iter().all(|x| x.is_finite())
            ));
        }
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if (norm - 1.0).abs() > 1e-3 {
            return Err(anyhow!(
                "vector is not unit length after normalization: {norm}"
            ));
        }
        let long = "测".repeat(max_tokens.saturating_mul(2).max(2));
        self.embed(&[long.as_str()]).with_context(|| {
            format!(
                "inference at MAX_TOKENS={max_tokens} (does the model support that many positions?)"
            )
        })?;
        Ok(())
    }

    /// Whether a panic during inference poisoned any pooled session lock.
    pub fn session_poisoned(&self) -> bool {
        self.pool.session_poisoned()
    }

    pub fn health_check(&self) -> Result<(), String> {
        self.health.check(self.session_poisoned())
    }

    pub fn embed(&self, texts: &[&str]) -> Result<Embedded> {
        let r = self.embed_inner(texts);
        self.health.record(r.is_ok());
        r
    }

    fn embed_inner(&self, texts: &[&str]) -> Result<Embedded> {
        if texts.is_empty() {
            return Ok(Embedded {
                vectors: vec![],
                tokens: vec![],
            });
        }

        // One tokenization pass for the whole batch. The encodings are
        // padded to the global longest text, but the attention mask still
        // tells us each text's true (unpadded) length, which is what the
        // chunk planner uses. Padding is re-applied per run below, so a
        // run of short texts is never padded to a long text's length.
        // `to_vec()` here copies only pointers (`&str`), not the texts.
        let encs = self
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| anyhow!("tokenize: {e}"))?;
        let token_lens: Vec<usize> = encs
            .iter()
            .map(|e| e.get_attention_mask().iter().filter(|&&m| m == 1).count())
            .collect();

        let mut out: Vec<Option<Vec<f32>>> = vec![None; texts.len()];
        for chunk in chunk_by_token_budget(&token_lens, self.max_tokens_per_run) {
            let vectors = self.run_chunk(&encs, &token_lens, &chunk)?;
            for (idx, v) in chunk.into_iter().zip(vectors) {
                out[idx] = Some(v);
            }
        }
        let vectors = out
            .into_iter()
            .map(|v| v.ok_or_else(|| anyhow!("missing embedding for a chunked text")))
            .collect::<Result<Vec<_>>>()?;

        Ok(Embedded {
            vectors,
            tokens: token_lens,
        })
    }

    /// Run one ONNX session call for the texts in `chunk` (indices into
    /// `encs`), padding only to the longest text in this run.
    fn run_chunk(
        &self,
        encs: &[Encoding],
        token_lens: &[usize],
        chunk: &[usize],
    ) -> Result<Vec<Vec<f32>>> {
        let b = chunk.len();
        let s = chunk
            .iter()
            .map(|&i| token_lens[i])
            .max()
            .unwrap_or(0)
            .max(1);
        let mut ids = Vec::with_capacity(b * s);
        let mut mask = Vec::with_capacity(b * s);
        let mut types = Vec::with_capacity(b * s);
        for &i in chunk {
            let e = &encs[i];
            let len = token_lens[i];
            ids.extend(e.get_ids()[..len].iter().map(|&x| x as i64));
            ids.resize(ids.len() + (s - len), 0);
            mask.extend(std::iter::repeat_n(1i64, len));
            mask.resize(mask.len() + (s - len), 0);
            types.extend(e.get_type_ids()[..len].iter().map(|&x| x as i64));
            types.resize(types.len() + (s - len), 0);
        }

        let pooled = self.pool.acquire().map_err(oe)?;
        let mut session = pooled.session().map_err(oe)?;

        let ids_t = Tensor::from_array(([b, s], ids)).map_err(oe)?;
        let mask_t = Tensor::from_array(([b, s], mask)).map_err(oe)?;
        let outputs = if self.has_token_type {
            let types_t = Tensor::from_array(([b, s], types)).map_err(oe)?;
            session
                .run(ort::inputs![
                    "input_ids" => ids_t,
                    "attention_mask" => mask_t,
                    "token_type_ids" => types_t
                ])
                .map_err(oe)?
        } else {
            session
                .run(ort::inputs![
                    "input_ids" => ids_t,
                    "attention_mask" => mask_t
                ])
                .map_err(oe)?
        };

        let (shape, data) = outputs[self.output_name.as_str()]
            .try_extract_tensor::<f32>()
            .map_err(oe)?;
        cls_pool_normalize(data, shape, b, s, self.dim)
    }
}

/// CLS pooling (token 0 of each sequence) followed by L2 normalization.
pub fn cls_pool_normalize(
    data: &[f32],
    shape: &[i64],
    b: usize,
    s: usize,
    dim: usize,
) -> Result<Vec<Vec<f32>>> {
    if shape.len() != 3
        || shape[0] as usize != b
        || shape[1] as usize != s
        || shape[2] as usize != dim
    {
        return Err(anyhow!(
            "unexpected output shape {shape:?}, want [{b}, {s}, {dim}]"
        ));
    }

    let mut out = Vec::with_capacity(b);
    for i in 0..b {
        let start = i * s * dim;
        let mut v = data[start..start + dim].to_vec();
        l2_normalize(&mut v);
        out.push(v);
    }
    Ok(out)
}

pub fn l2_normalize(v: &mut [f32]) {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 1e-12 {
        v.iter_mut().for_each(|x| *x /= n);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_unit_length() {
        let mut v = vec![3.0, 4.0];
        l2_normalize(&mut v);
        assert!((v[0] - 0.6).abs() < 1e-6 && (v[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn normalize_zero_vector_is_safe() {
        let mut v = vec![0.0, 0.0];
        l2_normalize(&mut v);
        assert_eq!(v, vec![0.0, 0.0]);
    }

    #[test]
    fn cls_pool_takes_first_token() {
        // b=2, s=2, dim=2: [[1,0],[9,9]] and [[0,2],[9,9]]
        let data = [1.0, 0.0, 9.0, 9.0, 0.0, 2.0, 9.0, 9.0];
        let out = cls_pool_normalize(&data, &[2, 2, 2], 2, 2, 2).unwrap();
        assert_eq!(out, vec![vec![1.0, 0.0], vec![0.0, 1.0]]);
    }

    #[test]
    fn cls_pool_rejects_bad_shape() {
        assert!(cls_pool_normalize(&[0.0; 4], &[1, 2, 2], 1, 2, 3).is_err());
    }

    #[test]
    fn chunking_covers_every_text_once_within_budget() {
        // Lengths 100, 400, 300, 10 with a 500-position budget:
        // sorted 400, 300, 100, 10 -> [400] (2*400 > 500), [300]
        // (2*300 > 500), then [100, 10] (2*100 <= 500).
        let chunks = chunk_by_token_budget(&[100, 400, 300, 10], 500);
        assert_eq!(chunks, vec![vec![1], vec![2], vec![0, 3]]);
        let mut seen: Vec<usize> = chunks.concat();
        seen.sort_unstable();
        assert_eq!(seen, vec![0, 1, 2, 3]);
    }

    #[test]
    fn chunking_keeps_an_oversize_text_alone() {
        let chunks = chunk_by_token_budget(&[9000, 10], 8192);
        assert_eq!(chunks, vec![vec![0], vec![1]]);
    }

    #[test]
    fn chunking_single_run_when_it_fits() {
        let chunks = chunk_by_token_budget(&[5, 5, 5], 8192);
        assert_eq!(chunks, vec![vec![0, 1, 2]]);
    }

    #[test]
    fn chunking_isolates_one_long_text_from_many_short_ones() {
        // The case that OOM-killed the first benchmark: 1x512 tokens +
        // 255x12 tokens with budget 8192. Padded-position budgeting puts
        // the long text with only 15 short ones (16*512 = 8192) and the
        // remaining 240 short texts in a second run (240*12 = 2880),
        // instead of one 256x512 run (131072 positions).
        let mut lens = vec![512];
        lens.extend(std::iter::repeat_n(12, 255));
        let chunks = chunk_by_token_budget(&lens, 8192);
        assert_eq!(chunks.len(), 2);
        for c in &chunks {
            let longest = c.iter().map(|&i| lens[i]).max().unwrap();
            assert!(c.len() * longest <= 8192, "{c:?}");
        }
        let mut seen: Vec<usize> = chunks.concat();
        seen.sort_unstable();
        assert_eq!(seen, (0..256).collect::<Vec<_>>());
    }

    #[test]
    fn chunking_empty_input_is_empty() {
        assert!(chunk_by_token_budget(&[], 8192).is_empty());
    }
}
