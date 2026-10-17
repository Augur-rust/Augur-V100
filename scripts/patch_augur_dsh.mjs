/** Apply or revert the version-bound Augur adapter patch without touching user data. */
import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import {fileURLToPath} from 'node:url';
const argv=process.argv.slice(2),i=argv.indexOf('--install');
if(i<0||!argv[i+1])throw Error('Required: --install DSH_INSTALL_DIRECTORY [--revert|--check]');
const spec=JSON.parse(await fs.readFile(path.resolve(path.dirname(fileURLToPath(import.meta.url)),'../integrations/dsh/dsh-patch.json'),'utf8'));
const dir=path.join(path.resolve(argv[i+1]),'resources','app','node_modules','@deepseek-ai','dsh-llm-pi-ai');
const pkg=JSON.parse(await fs.readFile(path.join(dir,'package.json'),'utf8'));
if(pkg.version!==spec.packageVersion)throw Error('DSH adapter version is not supported by this patch: '+pkg.version);
const file=path.join(dir,'lib','index.js'),backup=file+'.before-augur-'+spec.before_sha256.slice(0,12);
const sha=s=>crypto.createHash('sha256').update(s).digest('hex');
const original=await fs.readFile(file,'utf8'),actual=sha(original),revert=argv.includes('--revert');
if(argv.includes('--check')){console.log(JSON.stringify({patched:actual===spec.after_sha256,supported:actual===spec.before_sha256||actual===spec.after_sha256,sha256:actual}));process.exit(actual===spec.after_sha256?0:1);}
const wanted=revert?spec.before_sha256:spec.after_sha256;
if(actual===wanted){console.log(JSON.stringify({changed:false,patched:!revert,sha256:actual}));process.exit(0);}
if(actual!==(revert?spec.after_sha256:spec.before_sha256) && !(actual===spec.previous_patch_sha256 && !revert))throw Error('Installed adapter changed; refusing to overwrite an unknown build');
let next;
if(revert){next=await fs.readFile(backup,'utf8');}
else{
 next=actual===spec.previous_patch_sha256 ? await fs.readFile(backup,'utf8') : original;
 if(sha(next)!==spec.before_sha256)throw Error('Original backup hash mismatch');
 for(const [a,b] of spec.replacements){if(next.split(a).length!==2)throw Error('Patch anchor mismatch');next=next.replace(a,b);}
 next=spec.prefix+next;
 try{await fs.writeFile(backup,actual===spec.previous_patch_sha256 ? await fs.readFile(backup,'utf8') : original,{flag:'wx'});}catch(e){if(e.code!=='EEXIST'||sha(await fs.readFile(backup,'utf8'))!==spec.before_sha256)throw e;}
}
if(sha(next)!==wanted)throw Error('Patch or backup hash verification failed');
if(sha(await fs.readFile(file,'utf8'))!==actual)throw Error('Adapter changed during patch preparation');
const temp=file+'.augur-'+crypto.randomUUID()+'.tmp';await fs.writeFile(temp,next,{flag:'wx'});await fs.rename(temp,file);
console.log(JSON.stringify({changed:true,patched:!revert,sha256:wanted,backup,restartDshRequired:true}));
