import {test} from 'node:test';
import assert from 'node:assert/strict';
import {emptySession, sessionReducer} from '../src/session.ts';
test('showcase starts and resets without a parked vehicle',()=>{
 assert.equal(emptySession.active,false);
 const active=sessionReducer(emptySession,{type:'arrival',at:1000});
 assert.deepEqual(sessionReducer(active,{type:'reset'}),emptySession);
});
test('sensor arrival starts a session; duplicate sensor events preserve start time',()=>{
 const active=sessionReducer(emptySession,{type:'arrival',at:1000});
 assert.equal(active.startedAt,1000);
 assert.equal(active.active,true);
 assert.equal(sessionReducer(active,{type:'arrival',at:2000}).startedAt,1000);
});
test('payment requires detected parking and departure is sensor controlled',()=>{
 assert.deepEqual(sessionReducer(emptySession,{type:'pay'}),emptySession);
 const active=sessionReducer(emptySession,{type:'arrival',at:1000});
 const paid=sessionReducer(active,{type:'pay'});
 assert.equal(paid.paid,true);
 assert.equal(paid.active,true);
 assert.deepEqual(sessionReducer(paid,{type:'pay'}),paid);
 const departed=sessionReducer(paid,{type:'departure'});
 assert.equal(departed.active,false);
 assert.equal(departed.receipt,true);
 const next=sessionReducer(departed,{type:'arrival',at:3000});
 assert.equal(next.paid,false);
 assert.equal(next.startedAt,3000);
});
