import * as T from './vendor/three/build/three.module.js';
export function arcSample(t,r){return {position:new T.Vector3(r*Math.cos(t),r*Math.sin(t),0),tangent:new T.Vector3(-Math.sin(t),Math.cos(t),0)};}
export function makeRotationGlyph(axis,color,length){
 const group=new T.Group();group.quaternion.setFromUnitVectors(new T.Vector3(0,0,1),axis.clone().normalize());
 const material=new T.MeshBasicMaterial({color,depthTest:false,depthWrite:false});
 const shaft=new T.Mesh(new T.CylinderGeometry(length*.012,length*.012,length*1.25,10),material);shaft.rotation.x=Math.PI/2;shaft.position.z=length*.225;group.add(shaft);
 group.add(new T.ArrowHelper(new T.Vector3(0,0,1),new T.Vector3(),length,color,length*.18,length*.095));
 // Increasing theta in XY follows the right-hand rule about +Z.
 const radius=length*.29,end=4.75,points=[];
 for(let i=0;i<=64;i++)points.push(arcSample(.3+(end-.3)*i/64,radius).position);
 const arc=new T.Mesh(new T.TubeGeometry(new T.CatmullRomCurve3(points),64,length*.014,8,false),new T.MeshBasicMaterial({color:0xc87916,depthTest:false,depthWrite:false}));group.add(arc);
 const tip=arcSample(end,radius),cone=new T.Mesh(new T.ConeGeometry(length*.060,length*.17,12),arc.material);
 cone.quaternion.setFromUnitVectors(new T.Vector3(0,1,0),tip.tangent);cone.position.copy(tip.position).addScaledVector(tip.tangent,length*.035);group.add(cone);
 group.add(new T.Mesh(new T.SphereGeometry(length*.043,12,8),material));
 group.traverse(o=>{o.renderOrder=100;if(o.material){o.material.depthTest=false;o.material.depthWrite=false;}});return group;
}
export function axisInReference(jointWorld,axis,referenceWorld){const relative=referenceWorld.clone().invert().multiply(jointWorld);return {origin:new T.Vector3().setFromMatrixPosition(relative),direction:axis.clone().transformDirection(relative)};}
