async function saveSwitches(){
  const payload={yolo:swYolo.checked,deepsort:swDeepSort.checked,rule_fusion:swRule.checked};
  const res=await fetch("/model_settings",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(payload)});
  const data=await res.json();
  if(data.switches){swYolo.checked=data.switches.yolo;swDeepSort.checked=data.switches.deepsort;swRule.checked=data.switches.rule_fusion;updateModeText(data.switches);}
}
function updateModeText(sw){
  let text="";
  if(!sw.yolo) text="CNN/VGG16 + LSTM";
  else if(sw.yolo && !sw.deepsort) text="CNN/VGG16 + LSTM + YOLO + Fall Detection";
  else text="CNN/VGG16 + LSTM + YOLO + DeepSORT + Fall Detection + Running Detection";
  if(sw.rule_fusion) text+=" + Rule Fusion";
  const el=document.getElementById("currentMode"); if(el) el.innerHTML=text;
}
async function loadStatus(){
  const res=await fetch("/model_settings"); const data=await res.json();
  if(document.getElementById("swYolo")){
    swYolo.checked=data.switches.yolo; swDeepSort.checked=data.switches.deepsort; swRule.checked=data.switches.rule_fusion;
    ["swYolo","swDeepSort","swRule"].forEach(id=>document.getElementById(id).addEventListener("change",saveSwitches));
  }
  updateModeText(data.switches);
}
loadStatus();
