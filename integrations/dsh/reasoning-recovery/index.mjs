/** Bounded same-turn recovery. No failed reasoning is added to model history. */
import { createUserMessage } from '@deepseek-ai/dsh-llm';
import { z } from 'zod';
import { SingleFlight, serialize } from './single-flight.mjs';
export const name = 'augur-reasoning-recovery';
export const inject = ['agents', 'sessions', 'sessionProjections', 'llm'];
export const REPETITION = 'Repeated reasoning detected; generation was stopped after a long exact repetition. No completion was committed.';
export const INCOMPLETE = 'Augur recovery ended without a complete answer or tool action.';
const KEY = 'augurReasoningRecovery';
const KIND = 'augur-reasoning-recovery';
const route = c => c?.provider === 'augur-local' && c?.model === 'quasar-27b';
const count = z.number().int().nonnegative();
const configSchema = z.object({provider:z.string(), model:z.string(), reasoningEffort:z.string().optional(), maxTokens:count.positive().optional(), temperature:z.number().optional(), stop:z.array(z.string()).optional()});
const recoverySchema = z.object({turn:count, step:count, attempt:count.min(1).max(2), base:configSchema});
const stateSchema = z.object({turn:count.nullable(), count:count.max(2), active:recoverySchema.nullable(), restore:recoverySchema.nullable()});
export function initialState() { return {turn:null, count:0, active:null, restore:null}; }
function controls(recovery) {
  return {reasoningEffort:recovery.attempt === 1 ? 'low' : 'off', maxTokens:Math.min(recovery.base.maxTokens ?? 249036, recovery.attempt === 1 ? 8192 : 16384)};
}
// Full original controls are kept durably, including across turn end/restart.
export function fold(state, event) {
  if(event.type === 'turn/start') return {...state, turn:event.data.turn, count:0, active:null};
  if(event.type === 'turn/end') return {...state, active:null};
  if(event.type === 'user/message' && event.data.source?.kind === KIND) {
    const parsed = recoverySchema.safeParse(event.data.source.recovery);
    if(!parsed.success || parsed.data.turn !== state.turn || parsed.data.attempt !== state.count+1) return state;
    return {...state, count:parsed.data.attempt, active:parsed.data, restore:parsed.data};
  }
  if(event.type === 'assistant/message' && state.active && event.data.turn === state.active.turn && event.data.step === state.active.step) return {...state, active:null};
  if(event.type === 'request/header' && !state.active && state.restore) return {...state, restore:null};
  return state;
}
export function isLoopFailure(failure) {
  if(failure.code === 'AUGUR_RECOVERY_INCOMPLETE') return true;
  if(!['PI_AI_ERROR','reasoning_repetition'].includes(failure.code)) return false;
  const text=String(failure.message).trim();
  if(text === REPETITION) return true;
  // pi-ai may stringify the HTTP error envelope and prefix its HTTP status.
  try { const body=JSON.parse(text.replace(/^422\s*:?\s*/,'')); return (body.error ?? body).code === 'reasoning_repetition' && (body.error ?? body).message === REPETITION; } catch { return false; }
}
function restoreControls(config, recovery) {
  if(!route(config)) return config;
  const forced=controls(recovery), result={...config};
  for(const field of ['reasoningEffort','maxTokens']) {
    // A deliberate model/config change by another policy wins.
    if(config[field] !== forced[field]) continue;
    if(recovery.base[field] === undefined) delete result[field]; else result[field]=recovery.base[field];
  }
  return result;
}
export function apply(ctx, config={}) {
  if(Object.keys(config).length) throw Error('augur-reasoning-recovery has no configurable overrides');
  ctx.sessionProjections.register({key:KEY,stateVersion:1,stateSchema,init:initialState,apply:fold});
  let disposed=false;
  const queue=new SingleFlight();
  ctx.effect(() => () => { disposed=true; queue.close(); });
  const stateOf=agent => ctx.sessionProjections.stateOf(agent.session,KEY);
  ctx.on('agent/request',async ({agent,turn,step,signal},next) => {
    const base=await next();
    if(disposed || signal.aborted || !route(base)) return base;
    const state=stateOf(agent);
    if(state.active?.turn === turn && state.active.step === step) return {...base,...controls(state.active)};
    return state.restore ? restoreControls(base,state.restore) : base;
  });
  ctx.on('agent/request-error',async ({agent,turn,step,provider,failure,signal},next) => {
    if(disposed || signal.aborted) return;
    const current=agent.session.requestHeader()?.config;
    if(provider !== 'augur-local' || !route(current) || !isLoopFailure(failure)) return next();
    const state=stateOf(agent);
    // Own terminal handling too: do not delegate into a blind transport retry.
    if(state.turn !== turn || state.count >= 2) return;
    const base=state.active?.base ?? (state.restore ? restoreControls(current,state.restore) : current);
    const recovery={turn,step,attempt:state.count+1,base:configSchema.parse(base)};
    const text=recovery.attempt === 1
      ? 'Harness recovery notice: The previous generation repeated reasoning without making progress and was stopped. Continue the SAME original user task using the existing conversation and completed tool results. Do not repeat prior checks or restart completed work. Choose reasonable assumptions, make a brief plan if needed, and take the next concrete permitted tool action or provide the requested deliverable. Preserve all user constraints. Do not claim tools ran unless their results are present.'
      : 'Harness final recovery notice: The previous recovery still did not complete a useful action. Continue the SAME user task now with reasoning disabled for this request. Produce the requested answer or one concrete permitted tool call using existing results. Do not repeat the analysis, redo completed actions, or claim unperformed work. If essential information is missing, state the specific blocker concisely. Preserve user constraints; this is the final automatic recovery attempt.';
    agent.session.append('user/message',createUserMessage({content:[{type:'text',text}],source:{kind:KIND,form:'notice',summary:recovery.attempt === 1 ? '检测到重复思考，正在原任务中恢复（1/2）' : '再次恢复任务（2/2），本次临时关闭思考',recovery}}),{surfaceOp:'append'});
    return {kind:'retry'};
  });
  ctx.on('llm/stream',async function* (options,next) {
    const agent=options.sessionId ? ctx.agents.get(options.sessionId) : undefined;
    const recovery=agent && route(options) ? stateOf(agent).active : null;
    let completedAction=false;
    const stream=route(options) ? serialize(queue,options,next) : next();
    for await(const chunk of stream) {
      if(chunk.type === 'block-end' && ((chunk.block.type === 'text' && chunk.block.text.trim()) || chunk.block.type === 'tool-call')) completedAction=true;
      if(recovery && chunk.type === 'finish' && !['error','aborted'].includes(chunk.reason.kind) && (chunk.reason.kind === 'max-tokens' || !completedAction)) {
        yield {type:'finish',reason:{kind:'error',failure:{code:'AUGUR_RECOVERY_INCOMPLETE',message:INCOMPLETE}}};
      } else yield chunk;
    }
  });
}
