/** Configure the installed DSH Beta using its documented profile and credential files. */
import fs from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import crypto from 'node:crypto';
import {createRequire} from 'node:module';
import {fileURLToPath,pathToFileURL} from 'node:url';
const args=Object.fromEntries(process.argv.slice(2).reduce((pairs,x,i,all)=>i%2?pairs:[...pairs,[x.replace(/^--/,''),all[i+1]]],[]));
const installRecovery=args['reasoning-recovery']!=='false';
const contextOnly=args['context-only']==='true';
for(const k of (contextOnly?['install','home']:['install','home','key-file']))if(!args[k])throw Error('Required: --install PATH --home PATH --key-file PATH');
const install=path.resolve(args.install),home=path.resolve(args.home),keyPath=contextOnly?null:path.resolve(args['key-file']);
const profilePath=path.join(home,'profiles','desktop','cordis.patch.yml'),credentialsPath=path.join(home,'.credentials.yaml');
const req=createRequire(path.join(install,'resources','app','package.json'));const YAML=req('yaml');
const profile=JSON.parse(await fs.readFile(path.resolve(path.dirname(fileURLToPath(import.meta.url)),'../integrations/dsh/augur-provider.json'),'utf8'));
const compaction=JSON.parse(await fs.readFile(path.resolve(path.dirname(fileURLToPath(import.meta.url)),'../integrations/dsh/augur-compaction.json'),'utf8'));
const {BasicCompactionEngine}=await import(pathToFileURL(path.join(install,'resources','app','node_modules','@deepseek-ai','dsh-compaction-basic','lib','index.js')).href);
const checkedCompaction=BasicCompactionEngine.Config(compaction);
if(checkedCompaction.modelPolicies?.[0]?.outputReservationTokens!==0)throw Error('Installed DSH does not support the Augur compaction reservation policy; no files changed');
const modulePath=path.join(install,'resources','app','node_modules','@deepseek-ai','dsh-llm-pi-ai','lib','index.js');
const {Config}=await import(pathToFileURL(modulePath).href);Config({providers:{'augur-local':profile}});
const before=await fs.readFile(profilePath,'utf8'),credentialBefore=contextOnly?null:await fs.readFile(credentialsPath,'utf8');
const doc=YAML.parseDocument(before),cred=contextOnly?null:YAML.parseDocument(credentialBefore);
if(doc.errors.length||cred?.errors.length||!YAML.isSeq(doc.contents))throw Error('Invalid existing DSH YAML; left unchanged');
function patch(id,update){const matches=doc.contents.items.filter(x=>x.get('id')===id);if(matches.length>1)throw Error('Ambiguous DSH row: '+id);let row=matches[0];if(!row){row=doc.createNode({id});doc.contents.add(row);}row.set('config',doc.createNode(update(row.get('config')?.toJSON()??{})));}
let configuredProfile;
patch('llm-pi-ai',c=>{
 const previous=c.providers?.['augur-local'];
 if(contextOnly&&!previous)throw Error('Augur provider is missing; run the full configuration first');
 let next=profile;
 if(contextOnly){
  const models=previous.models??[];
  if(models.filter(m=>m.id==='quasar-27b').length!==1)throw Error('Missing or ambiguous QUASAR model');
  next={...previous,models:models.map(m=>m.id==='quasar-27b'?{...m,contextWindow:262144,maxTokens:249036}:m)};
  Config({providers:{'augur-local':next}});
 }
 if(installRecovery && next.retryPolicy?.mode==='always')throw Error('Augur bounded recovery requires a finite provider retry policy; unbounded always retry is not supported');
 configuredProfile=next;
 return {...c,providers:{...c.providers,'augur-local':next}};
});
if(!contextOnly)patch('agent-default-model',c=>({...c,provider:'augur-local',model:'quasar-27b',reasoningEffort:c.provider==='augur-local' ? (c.reasoningEffort ?? 'off') : 'off'}));
function compactionConfig(c){
 const policies=c.modelPolicies??[];
 if(!Array.isArray(policies))throw Error('Invalid existing compaction policies');
 const matches=policies.filter(p=>p.provider==='augur-local'&&p.model==='quasar-27b');
 if(matches.length>1)throw Error('Ambiguous Augur compaction policies');
 const next={...c,auto:true,modelPolicies:[...policies.filter(p=>p.provider!=='augur-local'||p.model!=='quasar-27b'),compaction.modelPolicies[0]]};
 BasicCompactionEngine.Config(next);
 return next;
}
patch('compaction-basic',compactionConfig);
// Web presets isolate compaction below config.plugins. The host-level row is
// disabled in recent DSH versions, so configuring only that row has no effect.
// Preserve each effective preset's complete child list, including !!js nodes.
const presetDirectory=path.join(install,'resources','app','node_modules','@deepseek-ai','dsh-web-app','presets');
const configuredPresets=[];
for(const filename of ['standard.patch.yml','ptc.patch.yml','minimal.patch.yml','cordis.patch.yml']){
 let raw;try{raw=await fs.readFile(path.join(presetDirectory,filename),'utf8');}catch(error){if(error.code==='ENOENT')continue;throw error;}
 const builtin=YAML.parseDocument(raw);
 if(builtin.errors.length||!YAML.isSeq(builtin.contents))throw Error('Invalid shipped DSH preset: '+filename);
 for(const change of builtin.contents.items){
  const inserted=change.get('insert');if(!YAML.isSeq(inserted))continue;
  for(const declaration of inserted.items){
   if(declaration.get('name')!=='@deepseek-ai/dsh-agent-preset')continue;
   const id=declaration.get('id');
   const existing=doc.contents.items.filter(row=>row.get('id')===id);
   if(existing.length>1)throw Error('Ambiguous DSH preset override: '+id);
   const config=(existing[0]?.get('config')??declaration.get('config')).clone();
   const plugins=config.get('plugins');
   if(!YAML.isSeq(plugins))throw Error('Preset override must retain a complete plugins list: '+id);
   let count=0;
   function updateChildren(entries){
    for(const child of entries.items){
     if(!YAML.isMap(child))throw Error('Invalid preset child: '+id);
     if(child.get('name')==='@deepseek-ai/dsh-compaction-basic'){
      child.set('config',doc.createNode(compactionConfig(child.get('config')?.toJSON()??{})));count++;
     }
     if(child.get('group')===true&&YAML.isSeq(child.get('config')))updateChildren(child.get('config'));
    }
   }
   updateChildren(plugins);
   if(!count)continue;
   let row=existing[0];if(!row){row=doc.createNode({id});doc.contents.add(row);}
   row.set('config',config);configuredPresets.push({id,compactionProviders:count});
  }
 }
}
let recoveryInstallation=null;
if(installRecovery){
 const sourceDirectory=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'../integrations/dsh/reasoning-recovery');
 const files=['package.json','index.mjs','single-flight.mjs'];
 const sourceFiles=await Promise.all(files.map(async name=>({name,bytes:await fs.readFile(path.join(sourceDirectory,name))})));
 for(const name of ['dsh-llm','dsh-agent-loop','dsh-session-projection','dsh-llm-pi-ai']){
  const installed=JSON.parse(await fs.readFile(path.join(install,'resources/app/node_modules/@deepseek-ai',name,'package.json'),'utf8'));
  if(installed.version!=='0.2.1-alpha.2')throw Error('Unverified DSH recovery API version: '+name+' '+installed.version);
 }
 const digest=crypto.createHash('sha256');for(const file of sourceFiles)digest.update(file.name).update(file.bytes);
 const sha256=digest.digest('hex');
 // Content-addressed installation keeps the loaded old plugin intact during updates.
 const destination=path.join(install,'resources/app/node_modules/@augur','dsh-reasoning-recovery-'+sha256.slice(0,16));
 await fs.mkdir(destination,{recursive:true});
 for(const file of sourceFiles){const target=path.join(destination,file.name);try{await fs.writeFile(target,file.bytes,{flag:'wx'});}catch(error){if(error.code!=='EEXIST'||!file.bytes.equals(await fs.readFile(target)))throw error;}}
 const plugin=pathToFileURL(path.join(destination,'index.mjs')).href;
 const id='augur-reasoning-recovery';let found=null;
 for(const row of doc.contents.items){
  const inserts=row.get('insert');if(!YAML.isSeq(inserts))continue;
  for(const child of inserts.items)if(child.get('id')===id){if(found)throw Error('Ambiguous Augur recovery plugin');found=child;}
 }
 if(found){found.set('name',plugin);found.set('disabled',false);}
 else doc.contents.add(doc.createNode({insert:[{id,name:plugin}]}));
 recoveryInstallation={plugin,sha256,maxRecoveries:2,providerQueue:true};
}
if(!contextOnly){
if(cred.get('version')!==1)throw Error('Unsupported credential layout; left unchanged');
await fs.mkdir(path.dirname(keyPath),{recursive:true});
let key;try{key=(await fs.readFile(keyPath,'utf8')).trim();}catch(e){if(e.code!=='ENOENT')throw e;key=crypto.randomBytes(32).toString('base64url');await fs.writeFile(keyPath,key+'\n',{flag:'wx',mode:0o600});}
if(key.length<32||/\s/.test(key))throw Error('Invalid local key');
const saved=cred.getIn(['refs','AUGUR_LOCAL_API_KEY']);if(saved!==undefined&&saved!==key)throw Error('Existing Augur credential differs; no overwrite');
cred.setIn(['refs','AUGUR_LOCAL_API_KEY'],key);
}
const stamp=new Date().toISOString().replace(/[:.]/g,'-');
async function replaceWithBackup(file,old,text){if(text===old)return;if(await fs.readFile(file,'utf8')!==old)throw Error('DSH file changed concurrently: '+file);await fs.copyFile(file,file+'.before-augur-'+stamp,fs.constants.COPYFILE_EXCL);const temp=file+'.augur-'+stamp+'.tmp';await fs.writeFile(temp,text,{flag:'wx',mode:0o600});await fs.rename(temp,file);}
if(!contextOnly)await replaceWithBackup(credentialsPath,credentialBefore,String(cred));await replaceWithBackup(profilePath,before,String(doc));
console.log(JSON.stringify({configured:true,provider:'augur-local',model:'quasar-27b',baseURL:configuredProfile.baseURL,profilePath,keyFile:keyPath,credentialsPrinted:false,contextOnly,contextWindow:262144,maxTokens:249036,compressionThresholdTokens:249036,compressionAtRequestBoundary:true,configuredPresets,recoveryInstallation,restartDshRequired:true,backupsSuffix:'.before-augur-'+stamp}));
