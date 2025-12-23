/**
 * Indecimal mini RAG - Professional Document Intelligence
 * Clean, Corporate-grade UI
 */

import { useState, useEffect, useRef } from 'react';
import './App.css';
import './markdown.css';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// Header Component
function Header() {
  return (
    <header className="header">
      <div className="header-content">
        <div className="logo-section">
          <div className="logo-icon">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
              <path d="M21 16V8a2 2 0 00-1-1.73l-7-4a2 2 0 00-2 0l-7 4A2 2 0 003 8v8a2 2 0 001 1.73l7 4a2 2 0 002 0l7-4A2 2 0 0021 16z" />
              <polyline points="3.27,6.96 12,12.01 20.73,6.96" />
              <line x1="12" y1="22.08" x2="12" y2="12" />
            </svg>
          </div>
          <div className="logo-text">
            <h1>Indecimal <span className="logo-highlight">mini RAG</span></h1>
            <span className="logo-subtitle">Intelligent Document Assistant</span>
          </div>
        </div>
      </div>
    </header>
  );
}

// Source Card
function SourceCard({ source }) {
  const [expanded, setExpanded] = useState(false);
  const scorePercent = Math.round(source.score * 100);

  return (
    <div className="source-card">
      <div className="source-header" onClick={() => setExpanded(!expanded)}>
        <div className="source-file">
          <svg className="source-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
            <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z" />
            <polyline points="14,2 14,8 20,8" />
          </svg>
          <span className="source-name">{source.source}</span>
        </div>
        <div className="source-relevance">
          <div className="relevance-bar">
            <div className="relevance-fill" style={{ width: `${scorePercent}%` }} />
          </div>
          <span className="relevance-text">{scorePercent}%</span>
        </div>
        <button className="expand-btn">{expanded ? '−' : '+'}</button>
      </div>
      {expanded && source.preview && (
        <div className="source-preview">{source.preview}</div>
      )}
    </div>
  );
}

// Professional Markdown Renderer
const MarkdownRenderer = ({ content }) => {
  if (!content) return null;

  // Split content by double newlines for paragraphs/sections
  const sections = content.split(/\n\n+/);

  return (
    <div className="markdown-content">
      {sections.map((section, idx) => {
        // Headers (### Text)
        if (section.startsWith('###')) {
          return <h3 key={idx}>{section.replace(/^###\s+/, '')}</h3>;
        }
        if (section.startsWith('##')) {
          return <h2 key={idx}>{section.replace(/^##\s+/, '')}</h2>;
        }
        // Bold header
        if (section.startsWith('**') && !section.includes('\n')) {
          return <h4 key={idx}>{section.replace(/\*\*/g, '')}</h4>;
        }

        // Lists (- Item)
        if (section.trim().startsWith('- ') || section.trim().startsWith('* ')) {
          const items = section.split('\n').filter(line => line.trim().startsWith('- ') || line.trim().startsWith('* '));
          return (
            <ul key={idx}>
              {items.map((item, i) => (
                <li key={i}>
                  {parseBold(item.replace(/^[-*]\s+/, ''))}
                </li>
              ))}
            </ul>
          );
        }

        // Regular Paragraphs with Bold parsing
        return (
          <p key={idx}>
            {section.split('\n').map((line, i) => (
              <span key={i} className="line-block">
                {parseBold(line)}
              </span>
            ))}
          </p>
        );
      })}
    </div>
  );
};

// Helper to parse **bold** inside text
const parseBold = (text) => {
  const parts = text.split(/(\*\*.*?\*\*)/g);
  return parts.map((part, i) => {
    if (part.startsWith('**') && part.endsWith('**')) {
      return <strong key={i}>{part.slice(2, -2)}</strong>;
    }
    return part;
  });
};

// Chat Message
function ChatMessage({ message, isUser }) {
  return (
    <div className={`chat-message ${isUser ? 'user-message' : 'assistant-message'}`}>
      <div className="message-avatar">{isUser ? 'U' : 'AI'}</div>
      <div className="message-content">
        <div className="message-text">
          {isUser ? message.content : <MarkdownRenderer content={message.content} />}
        </div>
        {!isUser && message.sources && message.sources.length > 0 && (
          <div className="message-sources">
            <div className="sources-header">Sources Referenced</div>
            <div className="sources-list">
              {message.sources.map((src, idx) => (
                <SourceCard key={idx} source={src} />
              ))}
            </div>
          </div>
        )}
        {!isUser && message.metadata && (
          <div className="message-meta">
            <span>{message.metadata.processing_time_ms?.toFixed(0)}ms</span>
            <span>{(message.metadata.confidence * 100).toFixed(0)}% confidence</span>
          </div>
        )}
      </div>
    </div>
  );
}

// Loading
function LoadingIndicator() {
  return (
    <div className="loading-indicator">
      <div className="loading-spinner" />
      <span>Analyzing documents...</span>
    </div>
  );
}

// Suggestions
function Suggestions({ onSelect }) {
  const items = [
    "Summarize the key points",
    "What topics are covered?",
    "List important requirements",
    "Explain the main concepts"
  ];

  return (
    <div className="suggestions">
      <div className="suggestions-title">Try these queries</div>
      <div className="suggestions-list">
        {items.map((item, idx) => (
          <button key={idx} className="suggestion-btn" onClick={() => onSelect(item)}>
            {item}
          </button>
        ))}
      </div>
    </div>
  );
}

// Chat Interface
function ChatInterface({ messages, isLoading, onSendMessage, hasDocuments }) {
  const [input, setInput] = useState('');
  const endRef = useRef(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (input.trim() && !isLoading && hasDocuments) {
      onSendMessage(input);
      setInput('');
    }
  };

  return (
    <div className="chat-interface">
      <div className="chat-messages">
        {messages.length === 0 ? (
          <div className="welcome">
            <div className="welcome-icon">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                <path d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
              </svg>
            </div>
            <h2>Document Intelligence</h2>
            <p>Upload documents and ask questions to get accurate, source-grounded answers.</p>
            {hasDocuments ? (
              <Suggestions onSelect={(s) => setInput(s)} />
            ) : (
              <div className="upload-prompt">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                  <path d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                </svg>
                <span>Upload a document to get started</span>
              </div>
            )}
          </div>
        ) : (
          <>
            {messages.map((msg, idx) => (
              <ChatMessage key={idx} message={msg} isUser={msg.role === 'user'} />
            ))}
            {isLoading && <LoadingIndicator />}
            <div ref={endRef} />
          </>
        )}
      </div>

      <form className="chat-form" onSubmit={handleSubmit}>
        <div className="input-wrapper">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder={hasDocuments ? "Ask about your documents..." : "Upload a document first"}
            disabled={isLoading || !hasDocuments}
          />
          <button type="submit" disabled={!input.trim() || isLoading || !hasDocuments}>
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <line x1="22" y1="2" x2="11" y2="13" />
              <polygon points="22,2 15,22 11,13 2,9" />
            </svg>
          </button>
        </div>
      </form>
    </div>
  );
}

// Sidebar
function Sidebar({ onUploadFile, isUploading, uploadedFiles }) {
  const [dragActive, setDragActive] = useState(false);
  const fileInputRef = useRef(null);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(e.type === "dragenter" || e.type === "dragover");
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files?.[0]) {
      onUploadFile(e.dataTransfer.files[0]);
    }
  };

  return (
    <aside className="sidebar">
      <div className="sidebar-content">
        <div className="upload-section">
          <h3>Upload Documents</h3>
          <div
            className={`upload-zone ${dragActive ? 'active' : ''} ${isUploading ? 'loading' : ''}`}
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept=".txt,.md,.pdf,.docx"
              onChange={(e) => e.target.files?.[0] && onUploadFile(e.target.files[0])}
              hidden
            />
            {isUploading ? (
              <>
                <div className="upload-spinner" />
                <span>Processing...</span>
              </>
            ) : (
              <>
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                  <path d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                </svg>
                <span>{dragActive ? 'Drop here' : 'Click or drag file'}</span>
                <small>TXT, MD, PDF, DOCX</small>
              </>
            )}
          </div>
        </div>

        {uploadedFiles.length > 0 && (
          <div className="files-section">
            <h4>Your Documents ({uploadedFiles.length})</h4>
            <ul className="file-list">
              {uploadedFiles.map((file, idx) => (
                <li key={idx}>
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                    <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z" />
                    <polyline points="14,2 14,8 20,8" />
                  </svg>
                  <span>{file}</span>
                  <svg className="check" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <polyline points="20,6 9,17 4,12" />
                  </svg>
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>

      <div className="sidebar-footer">Indecimal mini RAG v1.0</div>
    </aside>
  );
}

// Error Toast
function ErrorToast({ message, onClose }) {
  useEffect(() => {
    const timer = setTimeout(onClose, 5000);
    return () => clearTimeout(timer);
  }, [onClose]);

  return (
    <div className="error-toast">
      <span>{message}</span>
      <button onClick={onClose}>×</button>
    </div>
  );
}

// App
function App() {
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const [error, setError] = useState(null);

  const uploadFile = async (file) => {
    setIsUploading(true);
    setError(null);
    try {
      const formData = new FormData();
      formData.append('file', file);
      const res = await fetch(`${API_BASE_URL}/api/v1/upload`, { method: 'POST', body: formData });
      if (!res.ok) throw new Error((await res.json()).detail || 'Upload failed');
      setUploadedFiles(prev => [...new Set([...prev, file.name])]);
    } catch (err) {
      setError(err.message);
    } finally {
      setIsUploading(false);
    }
  };

  const sendMessage = async (content) => {
    setMessages(prev => [...prev, { role: 'user', content }]);
    setIsLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE_URL}/api/v1/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: content, top_k: 5 })
      });
      if (!res.ok) throw new Error((await res.json()).detail || 'Query failed');
      const data = await res.json();
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: data.answer,
        sources: data.sources,
        metadata: { confidence: data.confidence, processing_time_ms: data.processing_time_ms }
      }]);
    } catch (err) {
      setError(err.message);
      setMessages(prev => prev.slice(0, -1));
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="app">
      <Header />
      <main className="main">
        <Sidebar onUploadFile={uploadFile} isUploading={isUploading} uploadedFiles={uploadedFiles} />
        <ChatInterface messages={messages} isLoading={isLoading} onSendMessage={sendMessage} hasDocuments={uploadedFiles.length > 0} />
      </main>
      {error && <ErrorToast message={error} onClose={() => setError(null)} />}
    </div>
  );
}

export default App;
