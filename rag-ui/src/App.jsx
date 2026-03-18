import { useState } from "react";
import ChatPanel from "./components/ChatPanel";
import DocumentPanel from "./components/DocumentPanel";

export default function App() {

const [docUrl,setDocUrl] = useState(null)
const [collectionName, setCollectionName] = useState(null);

return(
<>
<div className="layout">

<div className="left">

<div className="agent-panel">
<div className="agent-title">
🤖 Agentic AI RAG
</div>

<div className="agent-sub">
✨ Ask questions about your documents
</div>
</div>

<div className="chat">
<ChatPanel/>
</div>


</div>

<div className="docs">
<DocumentPanel docUrl={docUrl} setDocUrl={setDocUrl} setCollectionName={setCollectionName}/>
</div>

</div>

<style>{`

body{
margin:0;
height: 100%;
font-family:Inter,system-ui;
background:#020617;
color:white;
}

.layout {
  display: flex;
  height: 100vh;  
  overflow: hidden; 
}


.left{
width:38%;
display:flex;
flex-direction:column;
border-right:1px solid #1e293b;
height: 100%;
overflow: hidden;
}

.agent-panel{
padding:18px 16px;
border-bottom:1px solid #1e293b;
background:#020617;
}

.agent-title{
font-size:18px;
font-weight:600;
margin-bottom:4px;
}

.agent-sub{
font-size:14px;
color:#94a3b8;
}


.chat {
  flex: 1;              
  overflow-y: auto; 
  padding: 10px;
  scrollbar-width: none;        
  -ms-overflow-style: none;  
}

.chat::-webkit-scrollbar {
  display: none;              
}

.docs {
  width: 60%;
  height: 100%;
  overflow: hidden;
}

`}</style>
</>
)
}
