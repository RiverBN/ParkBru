import {useEffect,useState} from 'react';
export type Appearance='system'|'light'|'dark';
export function useAppearance(){
 const [appearance,setAppearance]=useState<Appearance>(()=>{
  try{const saved=localStorage.getItem('parkbru-appearance');return saved==='light'||saved==='dark'?saved:'system'}catch{return 'system'}
 });
 useEffect(()=>{
  const query=window.matchMedia('(prefers-color-scheme: dark)');
  const apply=()=>{
   const theme=appearance==='system'?(query.matches?'dark':'light'):appearance;
   document.documentElement.dataset.theme=theme;
   document.documentElement.style.colorScheme=theme;
   document.querySelector('meta[name="theme-color"]')?.setAttribute('content',theme==='dark'?'#090e1a':'#f4f7fa');
  };
  apply();
  try{localStorage.setItem('parkbru-appearance',appearance)}catch{/* Keep the setting in memory when storage is unavailable. */}
  query.addEventListener('change',apply);
  return()=>query.removeEventListener('change',apply);
 },[appearance]);
 return [appearance,setAppearance] as const;
}
