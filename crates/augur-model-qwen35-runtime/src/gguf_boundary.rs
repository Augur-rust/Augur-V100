//! Runtime adapters for the model crate's GGUF input/output boundaries.
//!
//! This module owns one descriptor-bound reader for a target-only request.  It
//! is intentionally CUDA-independent: a future CUDA layer executor can use the
//! same embedding rows and output-head contract without changing request
//! lifecycle semantics.

use crate::{PrefillToken, TargetOnlyError, TargetOnlyLogitHead};
use augur_model_qwen35::{
    GgufEmbeddingRowProvider, GgufManifest, GgufStreamedOutputHead, GgufTensorReader, Qwen35Config,
};
use std::path::Path;

/// A target-only boundary that shares one authenticated GGUF reader between
/// embedding lookup and the streamed final norm/output head.
pub struct GgufTargetOnlyBoundary<'a> {
    reader: GgufTensorReader<'a>,
    embedding: GgufEmbeddingRowProvider,
    output_head: GgufStreamedOutputHead,
}

impl<'a> GgufTargetOnlyBoundary<'a> {
    /// Open the standard Qwen35 embedding, final norm, and output matrix.
    ///
    /// The caller must have inspected the manifest first.  The reader then
    /// revalidates the file identity and content before every tensor read.
    pub fn for_qwen35(
        manifest: &'a GgufManifest,
        path: impl AsRef<Path>,
        config: &Qwen35Config,
        output_chunk_rows: usize,
    ) -> Result<Self, TargetOnlyError> {
        let embedding = GgufEmbeddingRowProvider::for_qwen35(manifest, config)?;
        let mut reader = manifest.open_tensor_reader(path)?;
        let output_head =
            GgufStreamedOutputHead::for_qwen35(&mut reader, config, output_chunk_rows)?;
        Ok(Self {
            reader,
            embedding,
            output_head,
        })
    }

    pub fn embedding(&self) -> &GgufEmbeddingRowProvider {
        &self.embedding
    }

    pub fn output_head(&self) -> &GgufStreamedOutputHead {
        &self.output_head
    }

    /// Read one token embedding and package it for [`TargetOnlySession`].
    pub fn prefill_token(&mut self, token_id: u32) -> Result<PrefillToken, TargetOnlyError> {
        let token_id_usize = usize::try_from(token_id).map_err(|_| TargetOnlyError::Invalid {
            field: "token_id".into(),
            detail: "token id does not fit the host index type".into(),
        })?;
        let row = self
            .embedding
            .read_row_with_bytes(&mut self.reader, token_id_usize)
            .map_err(|error| TargetOnlyError::Model(error.to_string()))?;
        Ok(PrefillToken {
            token_id,
            hidden: row.values,
        })
    }

    /// Read an embedding row while retaining the source payload for a caller
    /// that needs to include it in a diagnostic digest.
    pub fn embedding_row(
        &mut self,
        token_id: u32,
    ) -> Result<augur_model_qwen35::GgufEmbeddingRow, TargetOnlyError> {
        let token_id_usize = usize::try_from(token_id).map_err(|_| TargetOnlyError::Invalid {
            field: "token_id".into(),
            detail: "token id does not fit the host index type".into(),
        })?;
        self.embedding
            .read_row_with_bytes(&mut self.reader, token_id_usize)
            .map_err(|error| TargetOnlyError::Model(error.to_string()))
    }

    pub fn reader(&mut self) -> &mut GgufTensorReader<'a> {
        &mut self.reader
    }
}

impl TargetOnlyLogitHead for GgufTargetOnlyBoundary<'_> {
    fn logits(&mut self, hidden: &[f32]) -> Result<Vec<f32>, TargetOnlyError> {
        self.output_head
            .logits(&mut self.reader, hidden)
            .map_err(|error| TargetOnlyError::Logits(error.to_string()))
    }
}
