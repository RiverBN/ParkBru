export type Session = {active:boolean; startedAt:number|null; paid:boolean; receipt:boolean};
export type SessionEvent = {type:'arrival';at:number}|{type:'departure'}|{type:'pay'}|{type:'reset'};
export const emptySession:Session={active:false,startedAt:null,paid:false,receipt:false};
export function sessionReducer(state:Session,event:SessionEvent):Session{
  switch(event.type){
    case 'arrival':return state.active?state:{active:true,startedAt:event.at,paid:false,receipt:state.receipt};
    case 'departure':return {...emptySession,receipt:state.receipt};
    case 'pay':return state.active&&!state.paid?{...state,paid:true,receipt:true}:state;
    case 'reset':return {...emptySession};
  }
}
