# code/main.py
"""
Local entrypoint for running PersonaSupervisor in CLI mode
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


def main():
    console.rule("[bold cyan]Persona AI Local CLI[/bold cyan]")

    session_id = str(uuid.uuid4())[:8]
    console.print(f"[dim]Session ID:[/dim] {session_id}")

    state_manager = StateManager()

    retriever = get_retriever(fail_if_empty=True)
    supervisor = PersonaSupervisor(retriever=retriever)

    state = state_manager.load(session_id)
    if state is None:
        state = create_initial_state(session_id)
        state_manager.save(session_id, state)

    state, greet = supervisor.handle(state=state, user_input="")
    state_manager.save(session_id, state)
    if greet:
        console.print(f"\n[bold magenta]Assistant[/bold magenta]: {greet}\n")

    console.print("[green]System ready. Type 'exit' to quit.[/green]\n")

    while True:
        try:
            user_input = Prompt.ask("[bold blue]You[/bold blue]")

            if user_input.strip().lower() in {"exit", "quit"}:
                console.print("\n[bold yellow]Session ended.[/bold yellow]")
                state_manager.delete(session_id)
                break

            state, reply = supervisor.handle(state=state, user_input=user_input)

            state_manager.save(session_id, state)
            console.print(f"\n[bold magenta]Assistant[/bold magenta]: {reply}\n")

        except KeyboardInterrupt:
            console.print("\n[red]Interrupted by user.[/red]")
            break
        except Exception as e:
            console.print(f"[red]Unexpected error:[/red] {e}")


if __name__ == "__main__":
    main()