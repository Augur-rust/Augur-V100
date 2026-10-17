use augur_model_qwen35::{GgufAuditOptions, GgufPinnedArtifact};
use augur_quantize::{convert, plan, Result, DEFAULT_CHUNK_BYTES};
use std::path::PathBuf;
use std::ffi::OsString;

fn run_dflash(mode: &str, args: &mut impl Iterator<Item=OsString>) -> Result<()> {
    let source=PathBuf::from(args.next().ok_or("missing draft directory")?);
    let target=PathBuf::from(args.next().ok_or("missing target directory")?);
    let destination=if mode=="dflash-convert" {Some(PathBuf::from(args.next().ok_or("missing output directory")?))} else {None};
    let mut digest=None;let mut chunk=None;
    while let Some(flag)=args.next() {
        let value=args.next().ok_or("flag needs a value")?.into_string().map_err(|_|"non-UTF8 option")?;
        if flag=="--weights-sha256" && digest.is_none() {digest=Some(value);}
        else if flag=="--chunk-bytes" && chunk.is_none() {chunk=Some(value.parse::<usize>()?);}
        else {return Err("unknown or repeated DFlash option".into());}
    }
    let digest=digest.ok_or("--weights-sha256 is required to authenticate original weights")?;
    let chunk=chunk.unwrap_or(DEFAULT_CHUNK_BYTES);
    if !(128..=16*1024*1024).contains(&chunk) || chunk%128!=0 {return Err("invalid DFlash chunk size".into());}
    let value=if let Some(dest)=destination {augur_quantize::dflash::convert(&source,&target,&dest,&digest,chunk)?}
        else {serde_json::to_value(augur_quantize::dflash::plan(&source,&target,&digest,chunk)?)?};
    println!("{}",serde_json::to_string_pretty(&value)?);Ok(())
}


fn run() -> Result<()> {
    let mut args = std::env::args_os().skip(1);
    let mode = args
        .next()
        .ok_or("usage: augur-quantize <plan|convert> INPUT.gguf [OUTPUT_DIR] [--chunk-bytes N]; or <dflash-plan|dflash-convert> DRAFT_DIR TARGET_DIR [OUTPUT_DIR] --weights-sha256 HEX")?;
    if mode == "dflash-plan" || mode == "dflash-convert" {
        return run_dflash(mode.to_str().ok_or("invalid command")?, &mut args);
    }
    if mode != "plan" && mode != "convert" {
        return Err("expected plan, convert, dflash-plan or dflash-convert".into());
    }
    let input = PathBuf::from(args.next().ok_or("missing input GGUF")?);
    let destination = if mode == "convert" {
        Some(PathBuf::from(
            args.next().ok_or("missing output directory")?,
        ))
    } else {
        None
    };
    let mut chunk = DEFAULT_CHUNK_BYTES;
    if let Some(flag) = args.next() {
        if flag != "--chunk-bytes" {
            return Err("unknown argument".into());
        }
        chunk = args
            .next()
            .ok_or("missing chunk bytes")?
            .to_str()
            .ok_or("non-UTF8 chunk bytes")?
            .parse()?;
    }
    if args.next().is_some() {
        return Err("unexpected trailing arguments".into());
    }
    let result = if let Some(destination) = destination {
        convert(&input, &destination, chunk)?
    } else {
        let artifact = GgufPinnedArtifact::open(&input, GgufAuditOptions::default())?;
        serde_json::to_value(plan(&input, &artifact, chunk)?)?
    };
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("augur-quantize: {error}");
        std::process::exit(2);
    }
}
