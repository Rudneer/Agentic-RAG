import { useState } from "react";

export default function DocumentPanel({ docUrl, setDocUrl, setCollectionName }) {

  const [zoom, setZoom] = useState(1);
  const [filePath, setFilePath] = useState(null);
  const [fileType, setFileType] = useState(null);


  // 🔥 Loader state
  const [loading, setLoading] = useState(false);
  const [loadingText, setLoadingText] = useState("Analyzing document...");

  // 🔥 Messages
  const messages = [
    "Analyzing document...",
    "Running OCR...",
    "Extracting tables...",
    "Structuring data...",
    "Almost done..."
  ];

  let interval;

  const startLoaderMessages = () => {
    let i = 0;
    interval = setInterval(() => {
      setLoadingText(messages[i]);
      i = (i + 1) % messages.length;
    }, 2000);
  };

  const stopLoaderMessages = () => {
    clearInterval(interval);
  };

  const uploadFile = async (file) => {

    const formData = new FormData();
    formData.append("file", file);

    // 1️⃣ Upload file
    const res = await fetch("http://localhost:8000/upload", {
        method: "POST",
        body: formData
    });

    const data = await res.json();
    console.log(data)


    // 2️⃣ 🔥 SHOW DOCUMENT IMMEDIATELY
    // setDocUrl(data.url);
    setFilePath(data.file_path);
    // setCollectionName(data.collection_name);

    // 3️⃣ Start parsing in background
    try {
        setLoading(true);
        startLoaderMessages();

        const res2 = await fetch("http://localhost:8000/parse-document", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({
            file_path: data.file_path,
            collection_name: data.collection_name
        })
        });

        const result = await res2.json();

        // ✅ STORE IT HERE
        setCollectionName(result.collection_name);
        localStorage.setItem("collection_name", result.collection_name);

    } catch (err) {
        console.error(err);
    } finally {
        stopLoaderMessages();
        setLoading(false);
    }
  };


  const handleUpload = (e) => {
    const file = e.target.files[0];
    if (!file) return;

    // ✅ Detect type
    if (file.type.includes("image")) {
        setFileType("image");
    } else if (file.type === "application/pdf") {
        setFileType("pdf");
    }

    // ✅ Instant preview
    const localUrl = URL.createObjectURL(file);
    setDocUrl(localUrl);

    uploadFile(file);
  };



  const zoomIn = () => setZoom(z => z + 0.2);
  const zoomOut = () => setZoom(z => Math.max(0.4, z - 0.2));
  const resetZoom = () => setZoom(1);

  return (
    <>
      <div className="doc-container">

        <div className="doc-header">
          📄 Document Viewer

          <div className="controls">
            <button onClick={zoomOut}>➖</button>
            <button onClick={zoomIn}>➕</button>
            <button onClick={resetZoom}>🔄</button>

            <label className="upload">
              📤 Upload
              <input type="file" hidden onChange={handleUpload} />
            </label>
          </div>
        </div>

        <div className="viewer">

            {/* 🔥 ADD THIS BLOCK HERE */}
            {loading && (
                <div className="overlay">
                <div className="glass-card">
                    <div className="loader"></div>
                    <p>{loadingText}</p>
                </div>
                </div>
            )}

            {!docUrl && (
                <div className="center no-doc">
                📂
                <p>Upload document or image</p>

                <label className="center-upload">
                    Select File
                    <input type="file" hidden onChange={handleUpload}/>
                </label>
                </div>
            )}

            {docUrl && (
                <div
                  className="preview"
                  style={{
                    width: "100%",
                    height: "100%",
                    opacity: loading ? 0.5 : 1
                  }}
                >

                  {fileType === "image" && (
                    <img
                      src={docUrl}
                      alt="doc"
                      style={{
                        width: "100%",
                        height: "100%",
                        objectFit: "contain",
                        transform: `scale(${zoom})`,
                        transformOrigin: "center"
                      }}
                    />
                  )}

                  {fileType === "pdf" && (
                    <iframe
                      src={`${docUrl}#toolbar=0&navpanes=0`}
                      title="pdf"
                      style={{
                        width: "100%",
                        height: "100%",
                        border: "none"
                      }}
                    />
                  )}
                </div>
            )}


        </div>

      </div>


      <style>{`

      .doc-container{
        height:100%;
        display:flex;
        flex-direction:column;
      }

      .doc-header{
        display:flex;
        justify-content:space-between;
        align-items:center;
        padding:16px;
        border-bottom:1px solid #1e293b;
        font-weight:600;
      }

      .controls{
        display:flex;
        gap:10px;
      }

      .controls button{
        background:#1e293b;
        border:none;
        color:white;
        padding:6px 10px;
        border-radius:6px;
        cursor:pointer;
      }

      .upload{
        background:#2563eb;
        padding:6px 10px;
        border-radius:6px;
        cursor:pointer;
      }

      .viewer{
        flex:1;
        overflow:hidden;
        display:flex;
        position:relative;
      }


      .no-doc{
        margin:auto;
        text-align:center;
      }

      .center{
        text-align:center;
        color:#64748b;
      }

      .center-upload{
        margin-top:10px;
        display:inline-block;
        background:#3b82f6;
        padding:8px 14px;
        border-radius:8px;
        cursor:pointer;
        color:white;
      }

      .preview{
        width:100%;
        height:100%;
        display:flex;
        justify-content:center;
        align-items:center;
      }


      /* 🔥 GLASS LOADER */

      .overlay{
        position:absolute;   /* 🔥 CHANGE THIS */
        top:0;
        left:0;
        width:100%;
        height:100%;

        backdrop-filter:blur(8px);
        background:rgba(0,0,0,0.4);

        display:flex;
        justify-content:center;
        align-items:center;

        z-index:10;
      }


      .glass-card{
        padding:30px 40px;
        border-radius:20px;
        background:rgba(255,255,255,0.08);
        backdrop-filter:blur(20px);
        border:1px solid rgba(255,255,255,0.2);
        text-align:center;
        color:white;
        font-size:18px;
      }

      .loader{
        width:50px;
        height:50px;
        border:4px solid rgba(255,255,255,0.2);
        border-top:4px solid white;
        border-radius:50%;
        animation:spin 1s linear infinite;
        margin:0 auto 15px;
      }

      @keyframes spin{
        to{transform:rotate(360deg);}
      }

      `}</style>
    </>
  );
}
