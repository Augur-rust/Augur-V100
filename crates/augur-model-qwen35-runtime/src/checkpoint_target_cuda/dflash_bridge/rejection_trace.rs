//! Explicitly enabled diagnostic evidence. Never changes proposal/verification.
//! Raw target rows after the first rejection are not canonical labels.
use super::*;
use std::{fs::OpenOptions,io::{BufWriter,Write},path::Path};
pub(super) struct Trace {writer:BufWriter<std::fs::File>,round:usize}
impl Trace {
    pub(super) fn from_env()->Result<Option<Self>> {
        let path=match std::env::var("AUGUR_SAMPLING_REJECTION_TRACE") {
            Err(std::env::VarError::NotPresent)=>return Ok(None),Ok(p)=>p,Err(e)=>return Err(e.into()),
        };
        let path=Path::new(&path);
        if !path.is_absolute(){return Err("rejection trace requires an absolute new file path".into());}
        let file=OpenOptions::new().create_new(true).write(true).open(path)?;
        Ok(Some(Self{writer:BufWriter::new(file),round:0}))
    }
    pub(super) fn record(&mut self,mut row:Value)->Result<()> {
        if self.round>=65536 {return Err("diagnostic rejection trace round budget exhausted".into());}
        row["schema"]=json!(if row["proposal_q"].is_null(){"quasar-sampling-rejection-trace-v1"}else{"quasar-probability-rejection-trace-v2"});
        row["round"]=json!(self.round);row["diagnostic_only"]=json!(true);
        row["post_first_rejection_labels_masked"]=json!(true);
        row["draft_scores_are_proposal_probabilities"]=json!(false);
        serde_json::to_writer(&mut self.writer,&row)?;self.writer.write_all(b"\n")?;self.writer.flush()?;
        self.round+=1;Ok(())
    }
}
pub(super) fn checked_slots(output_len:usize,rejection:Option<&Value>,width:usize)->Result<Vec<usize>> {
    if output_len>width || width<2 || width>8 {return Err("audit width/output length".into());}
    let checked=if let Some(r)=rejection {
        let slot=r["offset"].as_u64().ok_or("audit rejection offset missing")? as usize;
        if slot!=output_len || slot==0 || slot>=width {return Err("audit rejection prefix mismatch".into());}slot
    }else{output_len.saturating_sub(1)};
    Ok((1..=checked).collect())
}
#[cfg(test)] mod tests {
    use super::*;
    #[test] fn masks_unchecked_tail_at_every_first_rejection() {
        for slot in 1..8 {let r=json!({"offset":slot});assert_eq!(checked_slots(slot,Some(&r),8).unwrap(),(1..=slot).collect::<Vec<_>>());}
    }
    #[test] fn excludes_anchor_and_censored_rows_at_eos_or_length() {
        for n in 0..=8 {assert_eq!(checked_slots(n,None,8).unwrap(),(1..n).collect::<Vec<_>>());}
        assert!(checked_slots(2,Some(&json!({"offset":3})),8).is_err());
    }
}
