import { readdir, readFile, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
const assets=(await readdir('dist/assets')).map(name=>`/assets/${name}`);
const shell=['/','/index.html','/profile.png','/favicon-16.png','/favicon-32.png','/apple-touch-icon.png','/parkbru-icon-192.png','/parkbru-icon-512.png','/parkbru-maskable-512.png','/manifest.webmanifest',...assets];
const hash=createHash('sha256');
for(const url of shell){hash.update(await readFile(url==='/'?'dist/index.html':`dist${url}`));}
const cache='parkbru-'+hash.digest('hex').slice(0,12);
await writeFile('dist/sw.js',`const CACHE=${JSON.stringify(cache)};
const SHELL=${JSON.stringify(shell)};
self.addEventListener('install',event=>{event.waitUntil(caches.open(CACHE).then(cache=>cache.addAll(SHELL)));self.skipWaiting()});
self.addEventListener('activate',event=>{event.waitUntil(caches.keys().then(keys=>Promise.all(keys.filter(key=>key.startsWith('parkbru-')&&key!==CACHE).map(key=>caches.delete(key)))));self.clients.claim()});
self.addEventListener('fetch',event=>{if(event.request.method!=='GET'||new URL(event.request.url).origin!==self.location.origin)return;event.respondWith(fetch(event.request).then(response=>{if(response.ok){const copy=response.clone();caches.open(CACHE).then(cache=>cache.put(event.request,copy))}return response}).catch(()=>caches.match(event.request).then(hit=>hit||(event.request.mode==='navigate'?caches.match('/'):Response.error()))))});
`);
console.log(`PWA shell prepared: ${shell.length} files, ${cache}`);
