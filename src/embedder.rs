use anyhow::{anyhow, Context, Result};
use ort::environment::Environment;
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;
use std::path::Path;
use std::sync::Mutex;
use tokenizers::{PaddingParams, PaddingStrategy, Tokenizer, TruncationParams};

pub struct Embedder {
    _env: Environment,
    session: Mutex<Session>,
    tokenizer: Tokenizer,
    pub dim: usize,
    has_token_type: bool,
    output_name: String,
}

pub struct Embedded {
    pub vectors: Vec<Vec<f32>>,
    /// Non-padding token count per input text.
    pub tokens: Vec<usize>,
}

fn oe<E: std::fmt::Display>(e: E) -> anyhow::Error {
    anyhow!("{e}")
}

impl Embedder {
    pub fn load(model_dir: &Path, threads: usize, dim: usize, max_len: usize) -> Result<Self> {
        let mut tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json"))
            .map_err(|e| anyhow!("load tokenizer: {e}"))?;
        tokenizer
            .with_truncation(Some(TruncationParams { max_length: max_len, ..Default::default() }))
            .map_err(|e| anyhow!("truncation: {e}"))?;
        tokenizer.with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::BatchLongest,
            pad_id: 0,
            pad_token: "[PAD]".into(),
            ..Default::default()
        }));

        let env = ort::init().with_name("embed-svc").build().map_err(oe)?;
        let session = Session::builder(&env)
            .map_err(oe)?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(oe)?
            .with_intra_threads(threads)
            .map_err(oe)?
            .commit_from_file(model_dir.join("model.onnx"))
            .map_err(oe)
            .context("load model.onnx")?;
        let has_token_type = session.inputs().iter().any(|i| i.name() == "token_type_ids");
        let output_name = session
            .outputs()
            .first()
            .map(|o| o.name().to_string())
            .ok_or_else(|| anyhow!("model has no outputs"))?;
        Ok(Self { _env: env, session: Mutex::new(session), tokenizer, dim, has_token_type, output_name })
    }

    pub fn embed(&self, texts: &[String]) -> Result<Embedded> {
        if texts.is_empty() {
            return Ok(Embedded { vectors: vec![], tokens: vec![] });
        }
        let encs = self
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| anyhow!("tokenize: {e}"))?;
        let b = encs.len();
        let s = encs[0].get_ids().len();
        let mut ids = Vec::with_capacity(b * s);
        let mut mask = Vec::with_capacity(b * s);
        let mut types = Vec::with_capacity(b * s);
        let mut tokens = Vec::with_capacity(b);
        for e in &encs {
            ids.extend(e.get_ids().iter().map(|&x| x as i64));
            mask.extend(e.get_attention_mask().iter().map(|&x| x as i64));
            types.extend(e.get_type_ids().iter().map(|&x| x as i64));
            tokens.push(e.get_attention_mask().iter().filter(|&&m| m == 1).count());
        }

        let mut session = self.session.lock().map_err(|_| anyhow!("session lock poisoned"))?;
        let ids_t = Tensor::from_array(([b, s], ids)).map_err(oe)?;
        let mask_t = Tensor::from_array(([b, s], mask)).map_err(oe)?;
        let outputs = if self.has_token_type {
            let types_t = Tensor::from_array(([b, s], types)).map_err(oe)?;
            session.run(ort::inputs!["input_ids" => ids_t, "attention_mask" => mask_t, "token_type_ids" => types_t]).map_err(oe)?
        } else {
            session.run(ort::inputs!["input_ids" => ids_t, "attention_mask" => mask_t]).map_err(oe)?
        };
        let (shape, data) = outputs[self.output_name.as_str()].try_extract_tensor::<f32>().map_err(oe)?;
        let vectors = cls_pool_normalize(data, shape, b, s, self.dim)?;
        Ok(Embedded { vectors, tokens })
    }
}

/// CLS pooling (token 0 of each sequence) followed by L2 normalization.
pub fn cls_pool_normalize(data: &[f32], shape: &[i64], b: usize, s: usize, dim: usize) -> Result<Vec<Vec<f32>>> {
    if shape.len() != 3 || shape[0] as usize != b || shape[1] as usize != s || shape[2] as usize != dim {
        return Err(anyhow!("unexpected output shape {shape:?}, want [{b}, {s}, {dim}]"));
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
}
