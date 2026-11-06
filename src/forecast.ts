// Illustrative demand profiles; replace with the parking forecast API in production.
const profiles=[
 [5,4,3,3,4,6,12,20,28,35,46,65,88,82,64,48,42,51,67,78,61,36,18,9],
 [7,5,4,4,5,8,15,29,43,51,56,64,73,61,49,43,54,72,87,91,80,57,31,15],
 [4,3,2,2,3,5,16,38,72,84,77,68,81,74,59,46,36,28,24,21,17,12,8,5],
];
const hourAt=(at:number)=>Number(new Intl.DateTimeFormat('en-GB',{timeZone:'Asia/Brunei',hour:'2-digit',hourCycle:'h23'}).format(at));
const labelAt=(at:number)=>new Intl.DateTimeFormat('en-GB',{timeZone:'Asia/Brunei',hour:'numeric',hour12:true}).format(at).replace(' ','').toLowerCase();
export function getForecast(at:number,area:number){
 const profile=profiles[area]??profiles[0];
 const hour=hourAt(at);
 const points=Array.from({length:7},(_,i)=>{
  const offset=i-2;
  const value=profile[(hour+offset+24)%24];
  return {label:labelAt(at+offset*3600000),value,current:offset===0,peak:value>=75};
 });
 const current=profile[hour];
 const future=points.slice(3);
 const best=future.reduce((a,b)=>a.value<=b.value?a:b);
 const peak=current>=75;
 const title=peak?'Peak occupancy':current>=50?'Moderate occupancy':'Quieter parking';
 const tip=best.value<current?`More spaces expected around ${best.label} (${best.value}% occupancy).`:`Current occupancy is ${current}%. ${peak?'Allow extra time to find a space.':'A good time to look for parking.'}`;
 const time=new Intl.DateTimeFormat('en-GB',{timeZone:'Asia/Brunei',hour:'2-digit',minute:'2-digit',hour12:false}).format(at);
 return {points,title,tip,current,peak,time};
}
