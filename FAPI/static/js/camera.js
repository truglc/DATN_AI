let stream=null,timer=null;
const video=document.getElementById("video"),canvas=document.getElementById("canvas"),ctx=canvas.getContext("2d");
const cfg=window.CAMERA_CONFIG||{sendFps:10,canvasW:224,canvasH:224,jpegQuality:.45};
function setStatus(on){camStatus.innerText=on?"LIVE":"OFF";camStatus.classList.toggle("off",!on);}
function setResult(d){
  const label=d.label||"UNKNOWN"; labelBox.innerText=label; labelBox.className="result neutral";
  if(label==="FIGHT") labelBox.className="result fight";
  else if(label==="FALL_DETECTED"){labelBox.className="result fall";labelBox.innerText="FALL DETECTED";}
  else if(label==="RUNNING_ABNORMAL"){labelBox.className="result running";labelBox.innerText="RUNNING ABNORMAL";}
  else if(label.includes("NO FIGHT")) labelBox.className="result nofight";
  else if(label.includes("LOADING")) labelBox.className="result loading";
  fusionScore.innerText=Number(d.fusion_score||0).toFixed(2); lstmScore.innerText=Number(d.lstm_score||0).toFixed(2); ruleScore.innerText=Number(d.rule_score||0).toFixed(2);
  personCount.innerText=d.person_count||0; fallValue.innerText=d.fall_detected?"DETECTED":"OFF"; runningValue.innerText=d.running_abnormal?"DETECTED":"OFF";
  fpsValue.innerText=Number(d.fps||0).toFixed(1); latencyValue.innerText=Number(d.latency_ms||0).toFixed(0)+" ms"; seqValue.innerText=`${d.sequence_len||0}/${d.required_sequence||20}`;
}
async function sendFrame(){
 if(!stream) return; ctx.drawImage(video,0,0,cfg.canvasW,cfg.canvasH);
 const image=canvas.toDataURL("image/jpeg",cfg.jpegQuality);
 try{const res=await fetch("/predict_frame",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({image})}); const data=await res.json(); if(!data.error)setResult(data);}catch(e){console.error(e);}
}
btnStart.onclick=async()=>{stream=await navigator.mediaDevices.getUserMedia({video:{width:640,height:480},audio:false});video.srcObject=stream;setStatus(true);timer=setInterval(sendFrame,Math.max(80,Math.floor(1000/cfg.sendFps)));};
btnStop.onclick=()=>{if(timer)clearInterval(timer);timer=null;if(stream)stream.getTracks().forEach(t=>t.stop());stream=null;video.srcObject=null;setStatus(false);};
btnReset.onclick=async()=>{await fetch("/reset_camera_ai",{method:"POST"});setResult({label:"READY",fusion_score:0,lstm_score:0,rule_score:0,person_count:0,fps:0,latency_ms:0,sequence_len:0,required_sequence:20,fall_detected:false,running_abnormal:false});};
