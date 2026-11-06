import {existsSync, readFileSync} from 'node:fs';
import {dirname, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
import {defineConfig} from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';

const projectRoot=dirname(fileURLToPath(import.meta.url));
const httpsFiles={
  cert:resolve(projectRoot,'certs/parkbru.pem'),
  key:resolve(projectRoot,'certs/parkbru-key.pem'),
};
const detectorProxy={'/api':{target:'http://127.0.0.1:8765',changeOrigin:true}};

export default defineConfig(({mode})=>{
  let https: {cert:Buffer;key:Buffer}|undefined;
  if(mode==='https'){
    if(!existsSync(httpsFiles.cert)||!existsSync(httpsFiles.key)){
      throw new Error('Local HTTPS files are missing. Create certs/parkbru.pem and certs/parkbru-key.pem with mkcert; see README.md.');
    }
    https={cert:readFileSync(httpsFiles.cert),key:readFileSync(httpsFiles.key)};
  }
  return {
    plugins:[react(),tailwindcss()],
    server:{host:'0.0.0.0',https,proxy:detectorProxy},
    preview:{host:'0.0.0.0',https,proxy:detectorProxy},
  };
});
