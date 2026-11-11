import URDFViewer from './vendor/urdf-loader/src/urdf-viewer-element.js';
import * as T from 'three';
import {STLLoader} from 'three/examples/jsm/loaders/STLLoader.js';
import {makeRotationGlyph,axisInReference} from './axis-glyph.mjs';
import {PoseStream} from './pose-stream.mjs';
const $=id=>document.getElementById(id);
const fingerNames={thumb:'拇指',index:'食指',middle:'中指',ring:'无名指',pinky:'小指'};
const colors={thumb:0xb7795c,index:0x16887f,middle:0x467db5,ring:0x8f70a5,pinky:0xb7738c};
const labels={mcp_motor:'MCP 电机',pip_dip_motor:'PIP/DIP 联动电机',ip_motor:'IP 电机',cmc_flex_motor:'CMC 屈伸电机',cmc_abd_motor:'CMC 外展电机',cmc:'CMC',cmc_flex:'CMC 屈伸',cmc_abd:'CMC 外展',mcp:'MCP',mcp_abd:'MCP 外展',mcp_flex:'MCP 屈伸',pip:'PIP',dip:'DIP',ip:'IP'};
const short=n=>n.replace(/^[^_]+_/,'').replace(/_joint$/,'');
const standard=n=>fingerNames[n.split('_')[0]]+' · '+labels[short(n)];
let names={},motorLabels={},drafts={};const title=n=>names[n]||standard(n);
customElements.define('urdf-viewer',URDFViewer);
const viewer=$('model'),gizmos=new Map(),buttons=new Map(),meshMaterials=[];
let positions={},treePositions={},motorPositions={},dependentPositions={},definitions={},motorNames=[],independent=[],selected='index_mcp_flex_joint',mode='selected',length=.038,ready=false,busy=false,transparent=false,framePending=false,motion;
function fail(e){$('loading').classList.remove('hidden');$('loading').textContent='加载失败：'+e.message+'。请确认预览服务正在运行，再刷新页面。';console.error(e);}
function setBusy(value){busy=value;for(const id of ['positive','negative','joint-zero','reset','angle-slider'])$(id).disabled=!ready;$('inspector').setAttribute('aria-busy',String(value));}
function refreshAngles(){
 for(const [n,b] of buttons){b.querySelector('strong').textContent=names[n]||labels[short(n)];b.querySelector('span').textContent=(positions[n]||0).toFixed(3)+' rad';b.setAttribute('aria-pressed',String(n===selected));}
 const q=positions[selected]||0,target=motion?.target(selected)??q;$('angle-value').textContent=q.toFixed(3);$('angle-radians').textContent=(q*180/Math.PI).toFixed(1)+'°';$('angle-slider').value=target;
 $('angle-target').textContent=Math.abs(target-q)>.0005?'目标 '+target.toFixed(3)+' rad':'拖动实时联动';$('selected-title').textContent=title(selected);$('selected-name').textContent=selected;
 for(const[n,e]of gizmos)e.label.textContent=title(n)+' +';
}
function refreshOutputs(){
 $('motor-output').textContent=motorNames.map(n=>(motorLabels[n]||standard(n))+': '+(motorPositions[n]??0).toFixed(5)+' rad').join('\n');
 $('dependent-output').textContent=Object.entries(dependentPositions).map(([n,q])=>standard(n)+': '+q.toFixed(5)+' rad').join('\n');
}
function refreshName(){
 $('custom-name').value=Object.hasOwn(drafts,selected)?drafts[selected]:(names[selected]||'');
 $('name-status').textContent=Object.hasOwn(drafts,selected)?'此名称尚未保存。':`已命名 ${Object.keys(names).length} / 12 · 保存到本模型`;
}
function select(n){selected=n;refreshAngles();refreshName();updateAxes();if(ready)updateMaterials();}
function disposeGlyph(g){g.traverse(o=>{o.geometry?.dispose();if(o.material){for(const m of Array.isArray(o.material)?o.material:[o.material])m.dispose();}});}
function buildAxes(){
 for(const e of gizmos.values()){viewer.scene.remove(e.group);disposeGlyph(e.group);e.label.remove();}gizmos.clear();
 for(const n of independent){
  const definition=definitions[n],link=viewer.robot.links[definition.child];if(!link)throw new Error('URDF 中缺少指节刚体 '+definition.child);
  const axis=new T.Vector3(...definition.axis_local),local=makeRotationGlyph(axis,colors[n.split('_')[0]],length),group=new T.Group();group.add(local);viewer.scene.add(group);
  const label=document.createElement('button');label.className='axis-label';label.textContent=title(n)+' +';label.setAttribute('aria-label','选择轴 '+n);label.onclick=()=>select(n);$('axis-labels').append(label);
  gizmos.set(n,{group,label,link,axis});
 }updateAxes();
}
function updateAxes(){
 if(!ready)return;viewer.robot.updateWorldMatrix(true,true);
 for(const[n,e]of gizmos){e.link.getWorldPosition(e.group.position);e.link.getWorldQuaternion(e.group.quaternion);e.group.visible=mode==='all'||(mode==='selected'&&n===selected);e.label.classList.toggle('selected',n===selected);}
 const entry=gizmos.get(selected),v=axisInReference(entry.link.matrixWorld,entry.axis,viewer.robot.matrixWorld);
 for(const k of ['x','y','z']){$('axis-'+k).textContent=(Math.abs(v.direction[k])<.00005?0:v.direction[k]).toFixed(4);$('origin-'+k).textContent=(v.origin[k]*1000).toFixed(2);}
 $('local-axis').textContent=entry.axis.toArray().map(v=>Math.abs(v)<1e-10?'0':Number(v.toFixed(4))).join('  ');viewer.redraw();scheduleLabels();
}
function scheduleLabels(){if(framePending)return;framePending=true;requestAnimationFrame(()=>{framePending=false;projectLabels();});}
function projectLabels(){
 if(!ready)return;const w=$('stage').clientWidth,h=$('stage').clientHeight,placed=[];viewer.camera.updateMatrixWorld();
 for(const[n,e]of [...gizmos].sort(([a],[b])=>a===selected?-1:b===selected?1:0)){
  const show=e.group.visible&&$('show-labels').checked;e.label.hidden=!show;if(!show)continue;
  const p=e.axis.clone().multiplyScalar(length*1.14).applyMatrix4(e.link.matrixWorld).project(viewer.camera);
  if(p.z< -1||p.z>1||Math.abs(p.x)>1||Math.abs(p.y)>1){e.label.hidden=true;continue;}
  const x=(p.x*.5+.5)*w;let y=(-p.y*.5+.5)*h;
  for(let i=0;i<5&&placed.some(v=>Math.abs(v.x-x)<85&&Math.abs(v.y-y)<23);i++)y+=24;
  placed.push({x,y});e.label.style.left=Math.max(50,Math.min(w-50,x))+'px';e.label.style.top=Math.max(68,Math.min(h-90,y))+'px';
 }
}
function fit(){
 if(!ready)return;viewer.robot.updateWorldMatrix(true,true);const box=new T.Box3().setFromObject(viewer.robot),c=box.getCenter(new T.Vector3()),s=box.getSize(new T.Vector3()).length();
 viewer.camera.near=.001;viewer.camera.far=20;viewer.camera.fov=45;viewer.camera.position.copy(c).add(new T.Vector3(s*.72,s*.65,s*.82));viewer.controls.target.copy(c);viewer.camera.updateProjectionMatrix();viewer.controls.update();viewer.redraw();scheduleLabels();
}
function updateMaterials(){
 // 仅高亮选中指节定义的 child 刚体，不含后代刚体。
 const chosen=viewer.robot.links[definitions[selected].child];
 for(const item of meshMaterials){let node=item.mesh;while(node&&!node.isURDFLink)node=node.parent;const active=node===chosen;
  item.material.color.set(active?0x76aea7:0x9aafb8);item.material.transparent=transparent;item.material.opacity=transparent?(active?.75:.26):1;item.material.depthWrite=!transparent;item.material.needsUpdate=true;}
 viewer.redraw();
}
function applyPose(data,reset=false){
 if(!ready)return;
 $('status').classList.remove('error');
 if(reset)motion.reset();else motion.move(data);
 refreshAngles();
}
async function saveName(){
 const joint=selected,value=$('custom-name').value.trim();$('save-name').disabled=true;
 try{const res=await fetch('/api/names',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({joint,name:value})}),data=await res.json();if(!res.ok)throw new Error(data.error);
  names=data.names;delete drafts[joint];refreshAngles();if(selected===joint)refreshName();$('name-status').textContent='已保存到本地 · 已命名 '+Object.keys(names).length+' / 12';scheduleLabels();
 }catch(e){$('name-status').textContent='保存失败：'+e.message;}finally{$('save-name').disabled=false;}
}
async function init(){
 if(location.protocol==='file:')throw new Error('请先运行包内“启动预览.ps1”，再打开 http://localhost:8097/viewer/index.html');
 viewer.displayShadow=false;viewer.controls.minDistance=.01;viewer._updateEnvironment=()=>{};viewer.plane.visible=false;
 viewer.controls.enableDamping=true;viewer.controls.dampingFactor=.14;viewer.controls.rotateSpeed=1;viewer.controls.zoomSpeed=1.5;
 const[state,info,naming,roles]=await Promise.all([fetch('/api/state').then(r=>{if(!r.ok)throw new Error('闭链服务不可用');return r.json();}),fetch('./model_info.json').then(r=>r.json()),fetch('/api/names').then(r=>r.json()),fetch('/config/joint_roles.json').then(r=>{if(!r.ok)throw new Error('关节分类文件缺失');return r.json();})]);
 if(roles.kinematic_independent.length!==12||JSON.stringify(roles.kinematic_independent)!==JSON.stringify(state.joint_names))throw new Error('指节输入列表与闭环模型不一致');
 independent=state.joint_names;definitions=Object.fromEntries(state.joint_definitions.map(d=>[d.name,d]));positions=state.positions;treePositions=state.tree_positions;motorPositions=state.motor_positions;dependentPositions=state.dependent_positions;motorNames=state.motor_names;names=naming.names||{};motorLabels=naming.motor_names||{};refreshOutputs();$('joint-count').textContent=independent.length+' JOINTS';$('side-badge').textContent=info.side==='R'?'RIGHT HAND / 右手':'LEFT HAND / 镜像左手';$(info.side==='R'?'right-link':'left-link').classList.add('active');document.title=(info.side==='R'?'右手':'左手')+' · 真实指节旋转轴';
 motion=new PoseStream({positions,
  send:async(data,reset)=>{const res=await fetch(reset?'/api/reset':'/api/pose',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});return res.json();},
  onBusy:setBusy,
  onState:state=>{
   positions=state.positions;treePositions=state.tree_positions;motorPositions=state.motor_positions;dependentPositions=state.dependent_positions;viewer.setJointValues(treePositions);refreshAngles();refreshOutputs();updateAxes();
   $('status').classList.toggle('error',!!state.error);
   $('status').textContent=(state.error?'目标不可达，已恢复最后有效姿态：'+state.error+' · ':'实时闭环 · ')+'最大闭合误差 '+(state.gap_m*1000).toFixed(4)+' mm';
  },
  onError:e=>{$('status').classList.add('error');$('status').textContent='操作未完成：'+e.message;refreshAngles();}
 });
 for(const[finger,chinese]of Object.entries(fingerNames)){
  const section=document.createElement('section');section.className='finger-group';const heading=document.createElement('div');heading.className='finger-heading';
  const dot=document.createElement('i');dot.className='finger-dot';dot.style.background='#'+colors[finger].toString(16).padStart(6,'0');heading.append(dot,chinese);const tag=document.createElement('small');tag.textContent=finger.toUpperCase();heading.append(tag);const grid=document.createElement('div');grid.className='joint-grid';
  for(const n of independent.filter(n=>n.startsWith(finger+'_'))){const b=document.createElement('button');b.className='joint-button';b.setAttribute('aria-label',n);b.setAttribute('aria-pressed',String(n===selected));b.append(document.createElement('strong'),document.createElement('span'));b.onclick=()=>select(n);grid.append(b);buttons.set(n,b);}section.append(heading,grid);$('joint-list').append(section);
 }refreshAngles();refreshName();
 viewer.loadMeshFunc=(path,manager,done)=>new STLLoader(manager).load(path,g=>done(new T.Mesh(g,new T.MeshPhongMaterial({color:0x9aafb8,shininess:45}))),undefined,e=>{fail(new Error('网格加载失败'));done(null,e);});
 viewer.addEventListener('geometry-loaded',()=>{try{viewer.setJointValues(treePositions);viewer.robot.traverse(mesh=>{if(mesh.isMesh){mesh.material=mesh.material.clone();meshMaterials.push({mesh,material:mesh.material});}});viewer.ambientLight.intensity=1.8;ready=true;buildAxes();fit();updateMaterials();setBusy(false);$('loading').classList.add('hidden');$('status').textContent='闭环已连接 · 12 关节输入(rad) / 12 电机逆解输出';}catch(e){fail(e);}},{once:true});
 viewer.addEventListener('urdf-error',()=>fail(new Error('URDF 读取失败')));viewer.controls.addEventListener('change',scheduleLabels);new ResizeObserver(scheduleLabels).observe($('stage'));viewer.urdf='/hand_tree_only.urdf';
}
document.querySelectorAll('[data-mode]').forEach(b=>b.onclick=()=>{mode=b.dataset.mode;document.querySelectorAll('[data-mode]').forEach(x=>x.setAttribute('aria-pressed',String(x===b)));updateAxes();});
$('axis-size').oninput=()=>{length=Number($('axis-size').value)/1000;if(ready)buildAxes();};$('show-labels').onchange=scheduleLabels;$('fit').onclick=fit;
$('transparent').onclick=()=>{transparent=!transparent;$('transparent').setAttribute('aria-pressed',String(transparent));if(ready)updateMaterials();};
$('positive').onclick=()=>applyPose({[selected]:motion.target(selected)+.05});$('negative').onclick=()=>applyPose({[selected]:motion.target(selected)-.05});$('joint-zero').onclick=()=>applyPose({[selected]:0});$('reset').onclick=()=>applyPose({},true);
$('angle-slider').oninput=()=>applyPose({[selected]:Number($('angle-slider').value)});$('angle-slider').onchange=$('angle-slider').oninput;
$('custom-name').oninput=()=>{drafts[selected]=$('custom-name').value;$('name-status').textContent='尚未保存 · 切换关节会保留草稿';};$('custom-name').onkeydown=e=>{if(e.key==='Enter')saveName();};$('save-name').onclick=saveName;
$('export-names').onclick=async()=>{try{const data=await(await fetch('/api/names')).json(),url=URL.createObjectURL(new Blob([JSON.stringify({...data,independent_joints:independent.map(joint=>({joint,default_name:standard(joint),custom_name:data.names[joint]||''})),motor_actuators:motorNames.map(joint=>({joint,default_name:standard(joint),custom_name:data.motor_names[joint]||''}))},null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download='joint_and_motor_names.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}catch(e){$('name-status').textContent='导出失败：'+e.message;}};
window.addEventListener('focus',async()=>{try{const data=await(await fetch('/api/names')).json();names=data.names||{};motorLabels=data.motor_names||{};refreshAngles();refreshOutputs();refreshName();scheduleLabels();}catch{}});
window.addEventListener('resize',scheduleLabels);init().catch(fail);
