import { useState, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import "katex/dist/katex.min.css";

export default function ChatPanel({
  isProcessing,
  externalMessage,
  collectionName,
}) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");

  // external message
  useEffect(() => {
    if (externalMessage) {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: externalMessage },
      ]);
    }
  }, [externalMessage]);

  // typing effect
  const typeMessage = (text, callback) => {
    let i = 0;
    let current = "";

    const interval = setInterval(() => {
      current += text[i];
      i++;

      callback(current);

      if (i >= text.length) clearInterval(interval);
    }, 10);
  };

  const sendMessage = async (text) => {
    if (!text || isProcessing) return;

    setMessages((prev) => [
      ...prev,
      { role: "user", content: text },
      { role: "assistant", content: "Thinking..." },
    ]);

    setInput("");

    const res = await fetch("http://localhost:8000/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        query: text,
        collection_name:
          collectionName || localStorage.getItem("collection_name"),
      }),
    });

    const data = await res.json();

    typeMessage(data.answer, (partial) => {
      setMessages((prev) => {
        const updated = [...prev];

        updated[updated.length - 1] = {
          role: "assistant",
          content: partial,
          sources: data.sources,
          confidence: data.confidence,
        };

        return updated;
      });
    });
  };

  const handleKey = (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      sendMessage(input);
    }
  };

  return (
    <>
      <div className="chat-container">
        {/* messages */}
        <div className="messages">
          {messages.map((msg, i) => (
            <div key={i} className={`row ${msg.role}`}>
              
              {/* USER */}
              {msg.role === "user" && (
                <div className="bubble user">
                  {msg.content}
                </div>
              )}

              {/* ASSISTANT */}
              {msg.role === "assistant" && (
                <div className="bubble assistant">

                  <div className="markdown">
                    <ReactMarkdown
                      remarkPlugins={[remarkGfm, remarkMath]}
                      rehypePlugins={[rehypeKatex]}
                    >
                      {msg.content || ""}
                    </ReactMarkdown>
                  </div>

                  {msg.confidence && (
                    <div className="confidence">
                      Confidence: {(msg.confidence * 100).toFixed(0)}%
                    </div>
                  )}

                  {msg.sources && (
                    <div className="sources">
                      <div className="sources-title">
                        Sources
                      </div>

                      {msg.sources.map((s, idx) => (
                        <div key={idx} className="source-box">
                          {s.content}
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}

            </div>
          ))}
        </div>

        {/* input */}
        <div className="input-bar">
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKey}
            placeholder={
              isProcessing
                ? "Processing document..."
                : "Ask something..."
            }
            disabled={isProcessing}
          />

          <button
            onClick={() => sendMessage(input)}
            disabled={isProcessing}
          >
            🚀
          </button>
        </div>
      </div>

      <style>{`

.chat-container{
  height:100%;
  display:flex;
  flex-direction:column;
}

.messages{
  flex:1;
  overflow:auto;
  padding:12px;
  display:flex;
  flex-direction:column;
  gap:6px;
}

.row{
  display:flex;
}

.row.user{
  justify-content:flex-end;
}

.row.assistant{
  justify-content:flex-start;
}

.bubble{
  padding:8px 12px;
  border-radius:10px;
  font-size:14px;
  line-height:1.4;
}

.bubble.user{
  background:#2563eb;
  color:white;
  max-width:65%;
}

.bubble.assistant{
  background:#1e293b;
  color:#e2e8f0;
  max-width:90%;
}

/* markdown styling */

.markdown {
  overflow-x: auto;
}

.markdown p{
  margin:4px 0;
}

.markdown ul{
  margin:4px 0;
  padding-left:18px;
}

.markdown table {
  border-collapse: collapse;
  width: max-content;
  min-width: 100%;
}

.markdown th,
.markdown td {
  border: 1px solid #334155;
  padding: 6px 8px;
  white-space: nowrap;
}

.markdown::-webkit-scrollbar {
  height: 6px;
}

.markdown strong{
  color:white;
}

.markdown code{
  background:#0f172a;
  padding:2px 4px;
  border-radius:4px;
}

.confidence{
  margin-top:6px;
  font-size:12px;
  color:#94a3b8;
}

.sources{
  margin-top:8px;
}

.sources-title{
  font-size:12px;
  color:#94a3b8;
  margin-bottom:4px;
}

.source-box{
  background:#0f172a;
  padding:6px;
  border-radius:6px;
  margin-top:4px;
  font-size:12px;
}

.input-bar{
  display:flex;
  gap:8px;
  padding:10px;
  border-top:1px solid #1e293b;
}

.input-bar input{
  flex:1;
  padding:10px;
  border-radius:8px;
  border:none;
  background:#1e293b;
  color:white;
}

.input-bar button{
  background:#3b82f6;
  border:none;
  padding:8px 14px;
  border-radius:8px;
}

`}</style>
    </>
  );
}