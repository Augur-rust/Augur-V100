//! Reference continuation is teacher-only; never an input to propose_resident.
use super::*;
pub(super) struct TeacherReplay<'a> {pub case_id:&'a str,pub reference:&'a Run,pub stride:u64}
pub(super) fn sampled(case_id:&str,index:usize,stride:u64)->bool {
    assert!(matches!(stride,1|8));
    let digest=Sha256::digest(format!("quasar-complete-block-teacher-20261014|{case_id}|{index}").as_bytes());
    u64::from_le_bytes(digest[..8].try_into().unwrap())%stride==0
}
impl TeacherReplay<'_> {
    pub(super) fn teacher_path(&self,round_index:usize,output:&[u32],cursor:usize,path:&[u32])->Option<&[u32]> {
        let source=&self.reference.wire["rounds"][round_index];
        assert!(source.is_object(),"probe has no original baseline round");
        assert_eq!(source["output_start"].as_u64().unwrap() as usize,output.len());
        assert_eq!(source["start_consumed"].as_u64().unwrap() as usize,cursor);
        assert_eq!(ids(&source["proposal_path"]),path);
        assert_eq!(&self.reference.output[..output.len()],output,"baseline prefix diverged before probe");
        if !sampled(self.case_id,round_index,self.stride) || output.len()+8>self.reference.output.len(){return None;}
        let teacher=&self.reference.output[output.len()..output.len()+8];
        assert_eq!(teacher[0],path[0]);Some(teacher)
    }
    pub(super) fn check_round(&self,index:usize,value:&Value) {
        let source=&self.reference.wire["rounds"][index];
        for key in ["output_start","canonical_tokens","proposal_path","stop","start_consumed","end_consumed"] {
            assert_eq!(source[key],value[key],"complete teacher replay changed {key}");
        }
        assert_eq!(source["first_rejection"].is_null(),value["first_rejection"].is_null());
        if !source["first_rejection"].is_null(){for key in ["offset","correct_target_token","proposed_token"] {
            assert_eq!(source["first_rejection"][key],value["first_rejection"][key]);
        }}
    }
}
#[test]
fn complete_teacher_sample_is_deterministic_and_acceptance_independent() {
    assert!((0..32).all(|i|sampled("test_case",i,1)));
    let samples:Vec<_>=(0..100).filter(|&i|sampled("test_case",i,8)).collect();
    assert_eq!(samples,vec![1,22,26,36,39,44,48,49,82,84,86,91,98]);
    assert_eq!(samples,(0..100).filter(|&i|sampled("test_case",i,8)).collect::<Vec<_>>());
}
