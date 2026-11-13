// Stream only the latest targets; render only poses accepted by the closure solver.
export class PoseStream {
 constructor({positions,send,onState,onBusy=()=>{},onError=()=>{},schedule=callback=>requestAnimationFrame(callback),step=.02}) {
  Object.assign(this,{positions:{...positions},send,onState,onBusy,onError,schedule,step});
  this.targets={};this.resetPending=false;this.inFlight=false;this.scheduled=false;
 }
 target(name){return this.targets[name]??(this.resetPending?0:this.positions[name]??0);}
 move(values){Object.assign(this.targets,values);this._schedule();}
 reset(){this.targets={};this.resetPending=true;this._schedule();}
 _schedule(){
  if(this.inFlight||this.scheduled)return;
  this.scheduled=true;this.schedule(()=>{this.scheduled=false;this._pump();});
 }
 _prune(){for(const[n,q]of Object.entries(this.targets))if(Math.abs(q-(this.positions[n]??0))<1e-10)delete this.targets[n];}
 async _pump(){
  if(this.inFlight)return;
  this._prune();
  if(!this.resetPending&&!Object.keys(this.targets).length){this.onBusy(false);return;}
  const reset=this.resetPending;this.resetPending=false;const data={};
  if(!reset)for(const[n,q]of Object.entries(this.targets)){
   const current=this.positions[n]??0;data[n]=current+Math.max(-this.step,Math.min(this.step,q-current));
  }
  this.inFlight=true;this.onBusy(true);
  try {
   const state=await this.send(data,reset);
   if(!state.positions)throw new Error(state.error||'服务返回了无效姿态');
   this.positions={...state.positions};
   if(state.error&&!this.resetPending)this.targets={};
   this._prune();this.onState(state);
  } catch(e){this.targets={};this.onError(e);}
  finally {
   this.inFlight=false;
   const pending=this.resetPending||Object.keys(this.targets).length>0;
   this.onBusy(pending);if(pending)this._schedule();
  }
 }
}
