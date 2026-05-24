/** WebSocket events — spec §8.4 (ws://host:8000/ws/agent) */

export type ServerEvent =
  | { type: 'connected' }
  | { type: 'listening'; active: boolean }
  | { type: 'transcript'; role: 'user'; text: string; ts: number }
  | { type: 'agent_delta'; text: string }
  | { type: 'agent_message'; text: string; audio_url?: string }
  | { type: 'tool_status'; tool: string; status: 'running' | 'ok' | 'error'; detail?: string }
  | { type: 'thinking'; active: boolean }
  | { type: 'error'; message: string }

export type ClientEvent =
  | { type: 'listen_start' }
  | { type: 'listen_stop' }
  | { type: 'text'; content: string }

export function isServerEvent(value: unknown): value is ServerEvent {
  if (!value || typeof value !== 'object' || !('type' in value)) {
    return false
  }
  const t = (value as { type: string }).type
  return (
    t === 'connected'
    || t === 'listening'
    || t === 'transcript'
    || t === 'agent_delta'
    || t === 'agent_message'
    || t === 'tool_status'
    || t === 'thinking'
    || t === 'error'
  )
}

export function isClientEvent(value: unknown): value is ClientEvent {
  if (!value || typeof value !== 'object' || !('type' in value)) {
    return false
  }
  const t = (value as { type: string }).type
  return t === 'listen_start' || t === 'listen_stop' || t === 'text'
}
