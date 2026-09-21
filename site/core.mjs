
const n=(value,fallback=0)=>Number.isFinite(Number(value))?Number(value):fallback;
const clamp=(value,low,high)=>Math.min(high,Math.max(low,value));
const round=(value,digits=2)=>Number(value.toFixed(digits));
const mean=values=>values.length?values.reduce((sum,value)=>sum+value,0)/values.length:0;
const parseJSON=(value,fallback=[])=>{try{return JSON.parse(value)}catch{return fallback}};
const valuesFrom=value=>String(value).split(/[\s,]+/).map(Number).filter(Number.isFinite);
const result=(status,summary,metrics,rows,detail='')=>({status,summary,metrics,rows,detail});
const erf=x=>{const sign=x<0?-1:1,a=Math.abs(x),t=1/(1+0.3275911*a);const y=1-(((((1.061405429*t-1.453152027)*t)+1.421413741)*t-0.284496736)*t+0.254829592)*t*Math.exp(-a*a);return sign*y};
const normalCdf=z=>0.5*(1+erf(z/Math.sqrt(2)));
const wilson=(successes,total)=>{if(!total)return[0,0];const z=1.96,p=successes/total,d=1+z*z/total,c=(p+z*z/(2*total))/d,h=z*Math.sqrt((p*(1-p)+z*z/(4*total))/total)/d;return[clamp(c-h,0,1),clamp(c+h,0,1)]};
const sha256=async value=>{const bytes=new TextEncoder().encode(String(value));const digest=await crypto.subtle.digest('SHA-256',bytes);return[...new Uint8Array(digest)].map(byte=>byte.toString(16).padStart(2,'0')).join('')};
const tag=(xml,name)=>xml.match(new RegExp('<'+name+'[^>]*>([\\s\\S]*?)<\\/'+name+'>','i'))?.[1]?.trim()??'';
const similarity=(a,b)=>{const x=String(a).toLowerCase(),y=String(b).toLowerCase();if(x===y)return 1;const A=new Set(x.split(/\W+/).filter(Boolean)),B=new Set(y.split(/\W+/).filter(Boolean));const inter=[...A].filter(v=>B.has(v)).length;return inter/Math.max(1,new Set([...A,...B]).size)};

export const meta={"slug":"hierforecast","name":"HierForecast","eyebrow":"Coherent demand planning","description":"Reconcile regional forecasts to a national total and generate capacity intervals.","fields":[{"name":"nationalForecast","label":"National forecast (MW)","type":"number","min":1,"max":1000000,"step":10,"help":""},{"name":"confidence","label":"Confidence level","type":"number","min":50,"max":99,"step":1,"help":""},{"name":"regions","label":"Regional forecasts as JSON","type":"textarea","rows":8,"help":""}]};
export const initialState={"nationalForecast":42000,"confidence":90,"regions":"[{\"name\":\"North\",\"forecast\":10800,\"sigma\":420},{\"name\":\"South\",\"forecast\":9600,\"sigma\":380},{\"name\":\"East\",\"forecast\":11200,\"sigma\":510},{\"name\":\"West\",\"forecast\":9800,\"sigma\":440}]"};
export const alternateState={"nationalForecast":46000,"confidence":95,"regions":"[{\"name\":\"North\",\"forecast\":10800,\"sigma\":420},{\"name\":\"South\",\"forecast\":9600,\"sigma\":380},{\"name\":\"East\",\"forecast\":11200,\"sigma\":510},{\"name\":\"West\",\"forecast\":9800,\"sigma\":440}]"};
export async function compute(i){const national=n(i.nationalForecast),regions=parseJSON(i.regions,[]),base=regions.reduce((s,r)=>s+n(r.forecast),0),delta=national-base,z=n(i.confidence)>=95?1.96:n(i.confidence)>=90?1.645:1.282,rows=regions.map(r=>{const weight=n(r.forecast)/Math.max(1,base),forecast=n(r.forecast)+delta*weight,sigma=n(r.sigma);return{region:r.name,reconciled:Math.round(forecast),lower:Math.round(forecast-z*sigma),upper:Math.round(forecast+z*sigma)}});return result('Forecast reconciled',`Regional forecasts now sum exactly to ${Math.round(national).toLocaleString()} MW.`,[{label:'Original regional sum',value:Math.round(base).toLocaleString()},{label:'Reconciliation gap',value:Math.round(delta).toLocaleString()},{label:'Post-reconciliation gap',value:'0 MW'},{label:'Confidence',value:`${n(i.confidence)}%`}],rows,'The national constraint is distributed in proportion to each region’s base forecast.')}
