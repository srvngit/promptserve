<template>
  <div class="voice-app" data-od-id="voice-chat-app">
    <header class="voice-app__header" data-od-id="header">
      <slot name="header">
        <span class="font-semibold tracking-tight">Voice</span>
      </slot>
    </header>

    <main
      class="voice-app__thread"
      data-od-id="thread"
      tabindex="0"
      aria-label="Conversation"
    >
      <slot />
    </main>

    <footer class="voice-app__composer" data-od-id="composer">
      <slot name="composer">
        <p class="text-xs text-center text-[var(--muted)]">
          Press Enter to send · tap mic to speak
        </p>
      </slot>
    </footer>
  </div>
</template>
