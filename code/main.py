# code/main.py
"""
Local entrypoint for running PersonaSupervisor in CLI mode

Enterprise CLI baseline:
- Deterministic greet on session start via supervisor.handle(user_input="")
- Shows minimal debug state each turn (last_action / fsm_state / pending_slot)
- Provides 'reset' to wipe current session state (avoid stale state confusion)
"""

import uuid
from rich.console import Console
from rich.prompt import Prompt

from model.state_manager import StateManager
from model.conversation_state import ConversationState
from model.persona_supervisor import PersonaSupervisor
from service.local_vector_store import get_retriever

console = Console()


def create_initial_state(session_id: str) -> ConversationState:
    return ConversationState(
        session_id=session_id,
        persona_id="practical",
        context={},
    )


def _debug_line(state: ConversationState) -> str:
    ctx = state.context or {}
    fsm = (ctx.get("fsm_state") or "").strip() or "-"
    last_action = (getattr(state, "last_action", None) or "-").strip() if isinstance(getattr(state, "last_action", None), str) else (getattr(state, "last_action", None) or "-")
    pending = ctx.get("pending_slot")
    if isinstance(pending, dict):
        key = pending.get("key") or "-"
        opts = pending.get("options") or []
        try:
            nopts = len(opts)
        except Exception:
            nopts = "?"
        pending_txt = f"{key}({nopts})"
    else:
        pending_txt = "-"

    did_greet = bool(ctx.get("did_greet"))
    greet_streak = ctx.get("greet_streak", "-")
    return f"[dim]DEBUG[/dim] last_action={last_action} | fsm={fsm} | pending_slot={pending_txt} | did_greet={did_greet} | greet_streak={greet_streak}"


def main():
    console.rule("[bold cyan]Restbiz Persona AI Local CLI[/bold cyan]")

    session_id = str(uuid.uuid4())[:8]
    console.print(f"[dim]Session ID:[/dim] {session_id}")

    state_manager = StateManager()

    retriever = get_retriever(fail_if_empty=True)
    supervisor = PersonaSupervisor(retriever=retriever)

    state = state_manager.load(session_id)
    if state is None:
        state = create_initial_state(session_id)
        state_manager.save(session_id, state)
        console.print("[dim]State:[/dim] new session created")
    else:
        console.print("[yellow]Loaded existing session state (unexpected for new session_id).[/yellow]")

    # Deterministic greet on startup
    state, greet = supervisor.handle(state=state, user_input="")
    state_manager.save(session_id, state)

    if greet:
        console.print(f"\n[bold magenta]Assistant[/bold magenta]: {greet}\n")
    console.print(_debug_line(state))
    console.print("[green]System ready. Type 'exit' to quit. Type 'reset' to restart this session.[/green]\n")

    while True:
        try:
            user_input = Prompt.ask("[bold blue]You[/bold blue]")

            cmd = user_input.strip().lower()
            if cmd in {"exit", "quit"}:
                console.print("\n[bold yellow]Session ended.[/bold yellow]")
                state_manager.delete(session_id)
                break

            if cmd == "reset":
                console.print("\n[bold yellow]Resetting session state...[/bold yellow]")
                state = create_initial_state(session_id)
                state_manager.save(session_id, state)
                state, greet = supervisor.handle(state=state, user_input="")
                state_manager.save(session_id, state)
                if greet:
                    console.print(f"\n[bold magenta]Assistant[/bold magenta]: {greet}\n")
                console.print(_debug_line(state))
                console.print("")
                continue

            state, reply = supervisor.handle(state=state, user_input=user_input)
            state_manager.save(session_id, state)

            console.print(f"\n[bold magenta]Assistant[/bold magenta]: {reply}\n")
            console.print(_debug_line(state))
            console.print("")

        except KeyboardInterrupt:
            console.print("\n[red]Interrupted by user.[/red]")
            # keep state file for postmortem; do not delete automatically
            break
        except Exception as e:
            console.print(f"[red]Unexpected error:[/red] {e}")


if __name__ == "__main__":
    main()