import { useState, useEffect, useRef } from 'react'
import { Send, Bot, User, Activity, Server, Users, MessageCircle, Wifi, Loader2, RefreshCw } from 'lucide-react'
import './App.css'

interface Message {
  id: string
  content: string
  type: 'user' | 'assistant' | 'status' | 'error'
  timestamp: Date
}

interface AgentStatus {
  azure_agent_id: string | null
  thread_id: string | null
  available_remote_agents: number
  remote_agents: string[]
}

interface ApiStatus {
  message: string
  version: string
  status: string
  agent_status: AgentStatus
}

interface CurrentExecution {
  isExecuting: boolean
  agentName: string | null
  status: string
}

function App() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [apiStatus, setApiStatus] = useState<ApiStatus | null>(null)
  const [isConnected, setIsConnected] = useState(false)
  const [threadId, setThreadId] = useState<string | null>(null)
  const [currentExecution, setCurrentExecution] = useState<CurrentExecution>({
    isExecuting: false,
    agentName: null,
    status: ''
  })
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  useEffect(() => {
    fetchApiStatus()
  }, [])

  const fetchApiStatus = async () => {
    try {
      const response = await fetch('http://localhost:8083/')
      if (response.ok) {
        const data = await response.json()
        setApiStatus(data)
        setIsConnected(true)
        
        // Set thread_id if available from the API status
        if (data.agent_status && data.agent_status.thread_id) {
          setThreadId(data.agent_status.thread_id)
          console.log('Found existing thread ID:', data.agent_status.thread_id)
        }
      }
    } catch (error) {
      console.error('Failed to fetch API status:', error)
      setIsConnected(false)
    }
  }

  const startNewConversation = () => {
    setMessages([])
    setThreadId(null)
    setCurrentExecution({
      isExecuting: false,
      agentName: null,
      status: ''
    })
    
    // Update API status to reflect no active thread
    if (apiStatus) {
      setApiStatus({
        ...apiStatus,
        agent_status: {
          ...apiStatus.agent_status,
          thread_id: null
        }
      })
    }
    
    console.log('Started new conversation - thread_id reset')
  }

  const sendMessage = async () => {
    if (!input.trim() || isLoading) return

    const userMessage: Message = {
      id: crypto.randomUUID(),
      content: input.trim(),
      type: 'user',
      timestamp: new Date()
    }

    setMessages(prev => [...prev, userMessage])
    setInput('')
    setIsLoading(true)

    try {
      const requestBody = { 
        message: userMessage.content,
        thread_id: threadId
      }
      
      console.log('Sending request with thread_id:', threadId)
      
      const response = await fetch('http://localhost:8083/chat/stream', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody),
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const reader = response.body?.getReader()
      if (!reader) throw new Error('No reader available')

      let assistantMessage: Message | null = null

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        const chunk = new TextDecoder().decode(value)
        const lines = chunk.split('\n')

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6))
              
              if (data.type === 'response') {
                if (!assistantMessage) {
                  // Create the assistant message only when we get the first response
                  assistantMessage = {
                    id: crypto.randomUUID(),
                    content: data.content,
                    type: 'assistant',
                    timestamp: new Date()
                  }
                  setMessages(prev => [...prev, assistantMessage!])
                } else {
                  // Update existing message
                  setMessages(prev => 
                    prev.map(msg => 
                      msg.id === assistantMessage!.id 
                        ? { ...msg, content: data.content }
                        : msg
                    )
                  )
                }
                // Clear execution status when response is received
                setCurrentExecution({
                  isExecuting: false,
                  agentName: null,
                  status: ''
                })
              } else if (data.type === 'status') {
                if (!assistantMessage) {
                  // Create the assistant message for status
                  assistantMessage = {
                    id: crypto.randomUUID(),
                    content: data.content,
                    type: 'status',
                    timestamp: new Date()
                  }
                  setMessages(prev => [...prev, assistantMessage!])
                } else {
                  // Update existing message
                  setMessages(prev => 
                    prev.map(msg => 
                      msg.id === assistantMessage!.id 
                        ? { ...msg, content: data.content, type: 'status' }
                        : msg
                    )
                  )
                }
              } else if (data.type === 'agent_status') {
                // Handle agent execution status updates
                if (data.status_type === 'agent_start') {
                  setCurrentExecution({
                    isExecuting: true,
                    agentName: data.agent_name,
                    status: `Executing ${data.agent_name}`
                  })
                } else if (data.status_type === 'agent_complete') {
                  setCurrentExecution({
                    isExecuting: false,
                    agentName: data.agent_name,
                    status: `Completed ${data.agent_name}`
                  })
                  // Clear after a short delay
                  setTimeout(() => {
                    setCurrentExecution({
                      isExecuting: false,
                      agentName: null,
                      status: ''
                    })
                  }, 2000)
                }
              } else if (data.type === 'error') {
                if (!assistantMessage) {
                  // Create the assistant message for error
                  assistantMessage = {
                    id: crypto.randomUUID(),
                    content: data.content,
                    type: 'error',
                    timestamp: new Date()
                  }
                  setMessages(prev => [...prev, assistantMessage!])
                } else {
                  // Update existing message
                  setMessages(prev => 
                    prev.map(msg => 
                      msg.id === assistantMessage!.id 
                        ? { ...msg, content: data.content, type: 'error' }
                        : msg
                    )
                  )
                }
                // Clear execution status on error
                setCurrentExecution({
                  isExecuting: false,
                  agentName: null,
                  status: ''
                })
              }
            } catch (e) {
              console.error('Error parsing SSE data:', e)
            }
          }
        }
      }
      
      // After successful response, update thread_id if we didn't have one
      if (!threadId) {
        // Fetch the API status again to get the new thread_id
        setTimeout(async () => {
          await fetchApiStatus()
        }, 500) // Small delay to ensure backend has updated
      }
      
    } catch (error) {
      console.error('Error sending message:', error)
      const errorMessage: Message = {
        id: crypto.randomUUID(),
        content: 'Failed to send message. Please check your connection and try again.',
        type: 'error',
        timestamp: new Date()
      }
      setMessages(prev => [...prev, errorMessage])
    } finally {
      setIsLoading(false)
      // Clear execution status when request is completely done
      setCurrentExecution({
        isExecuting: false,
        agentName: null,
        status: ''
      })
    }
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <div className="header-content">
          <div className="header-title">
            <Bot className="header-icon" />
            <h1>Azure AI Foundry Routing Agent (A2A)</h1>
          </div>
          <div className="header-actions">
            <button 
              onClick={startNewConversation}
              className="new-conversation-btn"
              title="Start New Conversation"
            >
              <RefreshCw size={16} />
              New Chat
            </button>
            <div className="connection-status">
              <Wifi className={`status-icon ${isConnected ? 'connected' : 'disconnected'}`} />
              <span className={`status-text ${isConnected ? 'connected' : 'disconnected'}`}>
                {isConnected ? 'Connected' : 'Disconnected'}
              </span>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <div className="main-content">
        {/* Sidebar with Agent Status */}
        <aside className="sidebar">
          <div className="status-panel">
            <h3><Activity className="panel-icon" />AI Foundry Agent Status</h3>
            {isLoading && (
              <div className="thinking-animation">
                <Loader2 className="thinking-icon" />
                <span className="thinking-text">Processing your request</span>
                <div className="loading-dots">
                  <div className="loading-dot"></div>
                  <div className="loading-dot"></div>
                  <div className="loading-dot"></div>
                </div>
              </div>
            )}
            {currentExecution.isExecuting && (
              <div className="current-execution">
                <div className="execution-header">
                  <Loader2 className="execution-icon spinning" />
                  <span className="execution-text">Currently Executing</span>
                </div>
                <div className="execution-agent">
                  <strong>{currentExecution.agentName}</strong>
                </div>
              </div>
            )}
            {!currentExecution.isExecuting && currentExecution.agentName && (
              <div className="execution-complete">
                <div className="completion-header">
                  ✅ <span className="completion-text">Recently Completed</span>
                </div>
                <div className="completion-agent">
                  <strong>{currentExecution.agentName}</strong>
                </div>
              </div>
            )}
            {apiStatus && (
              <div className="status-grid">
                <div className="status-item">
                  <Server className="item-icon" />
                  <div>
                    <div className="item-label">API Version</div>
                    <div className="item-value">{apiStatus.version}</div>
                  </div>
                </div>
                <div className="status-item">
                  <Bot className="item-icon" />
                  <div>
                    <div className="item-label">Agent ID</div>
                    <div className="item-value">
                      {apiStatus.agent_status.azure_agent_id || 'Not initialized'}
                    </div>
                  </div>
                </div>
                <div className="status-item">
                  <MessageCircle className="item-icon" />
                  <div>
                    <div className="item-label">Thread ID</div>
                    <div className="item-value">
                      {threadId || (apiStatus.agent_status.thread_id ? apiStatus.agent_status.thread_id : 'No active thread')}
                    </div>
                  </div>
                </div>
                <div className="status-item">
                  <Users className="item-icon" />
                  <div>
                    <div className="item-label">Remote Agents</div>
                    <div className="item-value">{apiStatus.agent_status.available_remote_agents}</div>
                  </div>
                </div>
                {apiStatus.agent_status.remote_agents.length > 0 && (
                  <div className="remote-agents">
                    <h4>Available Agents:</h4>
                    {apiStatus.agent_status.remote_agents.map((agent, index) => (
                      <div key={index} className="remote-agent-item">
                        <div className="agent-dot"></div>
                        {agent}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>
        </aside>

        {/* Chat Area */}
        <main className="chat-container">
          <div className="messages">
            {messages.length === 0 && (
              <div className="welcome-message">
                <Bot className="welcome-icon" />
                <h2>Welcome to Azure AI Routing Agent using A2A</h2>
                <p>Ask me anything about sports results or general questions. I'll route your request to the appropriate specialized agent using A2A Agents.</p>
              </div>
            )}
            {messages.map((message) => (
              <div key={message.id} className={`message ${message.type}`}>
                <div className="message-avatar">
                  {message.type === 'user' ? <User /> : <Bot />}
                </div>
                <div className="message-content">
                  <div 
                    className="message-text"
                    dangerouslySetInnerHTML={
                      message.type === 'assistant' || message.type === 'status' 
                        ? { __html: message.content }
                        : undefined
                    }
                  >
                    {message.type === 'user' || message.type === 'error' ? message.content : null}
                  </div>
                  <div className="message-time">
                    {message.timestamp.toLocaleTimeString()}
                  </div>
                </div>
              </div>
            ))}
            <div ref={messagesEndRef} />
          </div>

          {/* Input Area */}
          <div className="input-container">
            <div className="input-wrapper">
              <textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Ask me about sports results or anything else..."
                className="message-input"
                rows={1}
                disabled={isLoading}
              />
              <button
                onClick={sendMessage}
                disabled={!input.trim() || isLoading}
                className="send-button"
              >
                <Send className="send-icon" />
              </button>
            </div>
          </div>
        </main>
      </div>
    </div>
  )
}

export default App
