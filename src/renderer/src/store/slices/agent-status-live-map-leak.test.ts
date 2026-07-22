/**
 * Memory-leak regression (issue #9872): the LIVE `agentStatusByPaneKey` map
 * must stay bounded.
 *
 * `agentStatusByPaneKey` is keyed by ephemeral paneKey (`${tabId}:${leafId}`,
 * leafId = a fresh UUID minted per pane, never reused). `setAgentStatus` writes
 * every ping's entry in place with a full spread copy
 * (`{ ...s.agentStatusByPaneKey, [paneKey]: entry }`) and keeps the row until a
 * pane/tab/worktree teardown event removes it. When a teardown event is missed
 * (agent killed without a Stop hook, pane/tab closed while its status lingered)
 * the row is orphaned and never removed, so on a long session orphans accumulate
 * without bound. Each entry is heavy (up-to-8KB lastAssistantMessage, up-to-16KB
 * interactivePrompt, prompt, up to 20 stateHistory rows, a sub-agent roster).
 * Because `setAgentStatus` spread-copies the WHOLE map on every ping, once the
 * map is large a single ping transiently doubles its bytes — the copy-on-write
 * spike in the crash bundle (heap pinned ~1965 MB, then +1.6 GB to the 3586 MB
 * old-space limit in one interval as a ping arrives -> -36861).
 *
 * #7528 capped the *retained* sibling map (`retainedAgentsByPaneKey`) but left
 * this *live* map uncapped, so the same accumulator reappeared here. The fix
 * caps the live map at MAX_LIVE_AGENT_STATUSES. A missed teardown can strand a
 * row in ANY state, so eviction keys off pane liveness, not `done`-ness:
 *  - a mounted tab has a rooted layout enumerating its live leaves, so a leaf
 *    missing from it is a PROVABLY DEAD pane -> evictable at any age;
 *  - an open pane's row is 'live' -> never evicted, even when its agent is idle;
 *  - anything else (rootless/empty-snapshot tab #2962, not-yet-hydrated tab, a
 *    runtime-attributed orchestration worker with no renderer tab) is UNPROVABLE
 *    -> kept while a fresh agent could still own it, shed only once idle past the
 *    stale window or by the hard-cap fallback that guarantees the bound.
 */
import { afterEach, describe, expect, it, vi } from 'vitest'
import {
  AGENT_STATUS_STALE_AFTER_MS,
  type ParsedAgentStatusPayload
} from '../../../../shared/agent-status-types'
import { MAX_LIVE_AGENT_STATUSES } from './agent-status'
import { createTestStore, makeTab, makeWorktree } from './store-test-helpers'
import type { AppState } from '../types'

// Worst-case per-entry payload the production type permits, so the leak's byte
// weight (not just entry count) is visible.
const BIG_ASSISTANT_MESSAGE = 'a'.repeat(8 * 1024)
const BIG_INTERACTIVE_PROMPT = 'q'.repeat(16 * 1024)

function seedWorktree(store: ReturnType<typeof createTestStore>): void {
  store.setState({
    repos: [
      { id: 'repo-1', path: '/repo', displayName: 'Repo', badgeColor: '#999999', addedAt: 1, kind: 'git' }
    ],
    worktreesByRepo: {
      'repo-1': [makeWorktree({ id: 'wt-1', repoId: 'repo-1', path: '/repo/wt-1' })]
    },
    tabsByWorktree: {
      'wt-1': [makeTab({ id: 'tab-live', worktreeId: 'wt-1' })]
    },
    // tab-live is mounted: a rooted layout with one live leaf. Its dead leaves are provable orphans.
    terminalLayoutsByTabId: {
      'tab-live': { root: { type: 'leaf', leafId: 'leaf-live' }, activeLeafId: 'leaf-live', expandedLeafId: null }
    }
  } as Partial<AppState>)
}

function donePayload(index: number): ParsedAgentStatusPayload {
  return {
    state: 'done',
    prompt: `prompt ${index}`,
    agentType: 'claude',
    lastAssistantMessage: BIG_ASSISTANT_MESSAGE,
    interactivePrompt: BIG_INTERACTIVE_PROMPT
  } as ParsedAgentStatusPayload
}

function workingPayload(index: number): ParsedAgentStatusPayload {
  return { state: 'working', prompt: `busy ${index}`, agentType: 'claude' } as ParsedAgentStatusPayload
}

function setAgentAt(
  store: ReturnType<typeof createTestStore>,
  paneKey: string,
  payload: ParsedAgentStatusPayload,
  updatedAt?: number
): void {
  const tabId = paneKey.slice(0, paneKey.indexOf(':'))
  store
    .getState()
    .setAgentStatus(paneKey, payload, undefined, updatedAt === undefined ? undefined : { updatedAt }, {
      tabId,
      worktreeId: 'wt-1'
    })
}

/** Flood `count` PROVABLY-DEAD leaves of the mounted tab-live (leaves absent from its rooted tree) —
 *  the dominant leak: panes churned and closed inside a long-lived open tab. */
function churnDeadLeaves(
  store: ReturnType<typeof createTestStore>,
  count: number,
  makePayload: (i: number) => ParsedAgentStatusPayload = donePayload
): void {
  for (let i = 0; i < count; i++) {
    setAgentAt(store, `tab-live:dead-${i}`, makePayload(i))
  }
}

/** Flood `count` fresh rows under tabs with no rooted layout — 'unprovable' rows (headless workers,
 *  replay of last-active state). Sweep 1 can't shed these while fresh, so they exercise the hard cap. */
function churnFreshUnprovable(store: ReturnType<typeof createTestStore>, count: number): void {
  for (let i = 0; i < count; i++) {
    setAgentAt(store, `gone-${i}:leaf-${i}`, workingPayload(i))
  }
}

describe('agentStatusByPaneKey stays bounded (leak regression #9872)', () => {
  afterEach(() => {
    vi.useRealTimers()
  })

  it('caps the live map at the limit, evicting the oldest dead-leaf orphans', () => {
    const store = createTestStore()
    seedWorktree(store)

    const total = MAX_LIVE_AGENT_STATUSES + 1500
    churnDeadLeaves(store, total)

    const live = store.getState().agentStatusByPaneKey
    // Bounded at the cap — NOT `total`. Without a cap this is `total` heavy entries, and every
    // subsequent ping spread-copies all of them (the OOM spike).
    expect(Object.keys(live).length).toBe(MAX_LIVE_AGENT_STATUSES)
    expect(live[`tab-live:dead-${total - 1}`]).toBeDefined()
    expect(live['tab-live:dead-0']).toBeUndefined()
  })

  it('never evicts a live pane\'s working agent', () => {
    const store = createTestStore()
    seedWorktree(store)
    setAgentAt(store, 'tab-live:leaf-live', workingPayload(0))

    churnDeadLeaves(store, MAX_LIVE_AGENT_STATUSES + 1500)

    expect(store.getState().agentStatusByPaneKey['tab-live:leaf-live']?.state).toBe('working')
  })

  it('never evicts a live pane\'s waiting or blocked row, even under the hard-cap fallback', () => {
    const store = createTestStore()
    seedWorktree(store)
    store.setState({
      terminalLayoutsByTabId: {
        'tab-live': { root: { type: 'leaf', leafId: 'leaf-live' }, activeLeafId: 'leaf-live', expandedLeafId: null },
        'tab-w': { root: { type: 'leaf', leafId: 'leaf-w' }, activeLeafId: 'leaf-w', expandedLeafId: null },
        'tab-b': { root: { type: 'leaf', leafId: 'leaf-b' }, activeLeafId: 'leaf-b', expandedLeafId: null }
      }
    } as Partial<AppState>)
    setAgentAt(store, 'tab-w:leaf-w', { state: 'waiting', prompt: 'needs input', agentType: 'claude' } as ParsedAgentStatusPayload)
    setAgentAt(store, 'tab-b:leaf-b', { state: 'blocked', prompt: 'perm prompt', agentType: 'claude' } as ParsedAgentStatusPayload)

    // Fresh unprovable rows force the hard-cap fallback, which sheds orphans of any state.
    churnFreshUnprovable(store, MAX_LIVE_AGENT_STATUSES + 1500)

    const live = store.getState().agentStatusByPaneKey
    expect(live['tab-w:leaf-w']?.state).toBe('waiting')
    expect(live['tab-b:leaf-b']?.state).toBe('blocked')
  })

  it('bounds the map when fresh unprovable rows dominate, without evicting a live pane', () => {
    const store = createTestStore()
    seedWorktree(store)
    setAgentAt(store, 'tab-live:leaf-live', donePayload(-1))

    // Missed-teardown orphans stuck in 'working' under no-layout tabs — a done-only cap could never
    // shed these, so the map would stay unbounded and re-OOM. The hard-cap fallback must bound it.
    churnFreshUnprovable(store, MAX_LIVE_AGENT_STATUSES + 200)

    const live = store.getState().agentStatusByPaneKey
    expect(Object.keys(live).length).toBe(MAX_LIVE_AGENT_STATUSES)
    expect(live['tab-live:leaf-live']?.state).toBe('done')
  })

  it('keeps fresh rows of rootless / empty-snapshot / no-renderer-tab panes (live agents, #2962)', () => {
    const store = createTestStore()
    seedWorktree(store)
    // bg-empty: inactive-worktree empty snapshot (root null, no bindings) — tab + PTY still live.
    // bg-bound: rootless snapshot that still holds PTY bindings.
    store.setState({
      tabsByWorktree: {
        'wt-1': [
          makeTab({ id: 'tab-live', worktreeId: 'wt-1' }),
          makeTab({ id: 'bg-empty', worktreeId: 'wt-1' })
        ]
      },
      terminalLayoutsByTabId: {
        'tab-live': { root: { type: 'leaf', leafId: 'leaf-live' }, activeLeafId: 'leaf-live', expandedLeafId: null },
        'bg-empty': { root: null, activeLeafId: null, expandedLeafId: null },
        'bg-bound': { root: null, activeLeafId: 'leaf-bound', expandedLeafId: null, ptyIdsByLeafId: { 'leaf-bound': 'pty-1' } }
      }
    } as Partial<AppState>)
    setAgentAt(store, 'bg-empty:leaf-bg', { state: 'waiting', prompt: 'needs input', agentType: 'claude' } as ParsedAgentStatusPayload)
    setAgentAt(store, 'bg-bound:leaf-bound', donePayload(-2))
    // A runtime-attributed orchestration worker with no renderer tab at all.
    setAgentAt(store, 'worker-tab:leaf-worker', workingPayload(-3))

    // Dominant leak (dead leaves) absorbs the overflow in sweep 1, so the fallback never runs.
    churnDeadLeaves(store, MAX_LIVE_AGENT_STATUSES + 200)

    const live = store.getState().agentStatusByPaneKey
    expect(live['bg-empty:leaf-bg']?.state).toBe('waiting')
    expect(live['bg-bound:leaf-bound']?.state).toBe('done')
    expect(live['worker-tab:leaf-worker']?.state).toBe('working')
  })

  it('evicts an idle unprovable orphan once past the stale window, keeping fresh ones', () => {
    // Fake timers so the freshness scheduler's Date.now() matches the synthetic timestamps below.
    vi.useFakeTimers()
    const late = new Date('2026-07-22T00:00:00.000Z').getTime()
    vi.setSystemTime(late)
    const store = createTestStore()
    seedWorktree(store)

    // A background row that hasn't pinged in > the stale window (its pane is likely gone).
    setAgentAt(store, 'gone-stale:leaf', workingPayload(0), late - AGENT_STATUS_STALE_AFTER_MS - 1)
    // Fill to the cap with fresh unprovable rows, then one more to force a single eviction.
    for (let i = 0; i < MAX_LIVE_AGENT_STATUSES; i++) {
      setAgentAt(store, `gone-fresh-${i}:leaf`, workingPayload(i), late)
    }

    const live = store.getState().agentStatusByPaneKey
    expect(Object.keys(live).length).toBe(MAX_LIVE_AGENT_STATUSES)
    // The stale orphan is shed first; the fresh ones (a live agent could still own them) are kept.
    expect(live['gone-stale:leaf']).toBeUndefined()
    expect(live['gone-fresh-0:leaf']?.state).toBe('working')
  })

  it('under cap: no eviction', () => {
    const store = createTestStore()
    seedWorktree(store)

    churnDeadLeaves(store, 10)

    expect(Object.keys(store.getState().agentStatusByPaneKey).length).toBe(10)
  })

  it('bumps agentStatusEpoch when eviction occurs', () => {
    const store = createTestStore()
    seedWorktree(store)

    churnDeadLeaves(store, MAX_LIVE_AGENT_STATUSES) // exactly at cap
    const epochBefore = store.getState().agentStatusEpoch

    setAgentAt(store, 'tab-live:dead-extra', donePayload(MAX_LIVE_AGENT_STATUSES)) // one more forces eviction

    const state = store.getState()
    expect(Object.keys(state.agentStatusByPaneKey).length).toBe(MAX_LIVE_AGENT_STATUSES)
    expect(state.agentStatusEpoch).toBeGreaterThan(epochBefore)
  })
})
