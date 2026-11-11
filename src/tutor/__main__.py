"""Command line entry point: ``python -m tutor``.

With no subcommand it validates the configuration and prints how it resolved
(Phase 2). Phase 9 adds ``stats`` and ``review``, which only touch the sqlite
database — no models are loaded, so they are instant.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from tutor.config import ConfigError, load_all
from tutor.errors import DeviceError
from tutor.core.prompts import (
    DEFAULT_SCENARIO,
    PromptError,
    available_scenarios,
    load_prompts,
)
from tutor.storage import Database
from tutor.storage.db import DOWN_THRESHOLD, UP_THRESHOLD
from tutor.ui.terminal import run_tutor

SCENARIOS_DIR_LABEL = "prompts/scenarios"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tutor",
        description="Tutor de ingles por voz (local, baixa latencia).",
    )
    parser.add_argument(
        "--config",
        default=None,
        metavar="PATH",
        help="caminho do config.yaml (padrao: ./config.yaml)",
    )
    parser.add_argument(
        "--levels",
        default=None,
        metavar="PATH",
        help="caminho do levels.yaml (padrao: ./levels.yaml)",
    )
    parser.add_argument(
        "--level",
        default="beginner",
        metavar="NAME",
        help="nivel de ingles: beginner | intermediate | advanced",
    )
    parser.add_argument(
        "--scenario",
        default=None,
        metavar="NAME",
        help="cenario de conversa, ex.: restaurant (prompts/scenarios/restaurant.md)",
    )
    parser.add_argument(
        "--accent",
        choices=("american", "british"),
        default=None,
        help="acento do TTS (padrao: o de config.yaml)",
    )
    parser.add_argument(
        "--list-levels",
        action="store_true",
        help="lista os niveis carregados e sai",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="so valida a configuracao e imprime a resolucao (nao abre audio)",
    )
    parser.add_argument(
        "--turns",
        type=int,
        default=None,
        metavar="N",
        help="encerra apos N turnos aceitos (padrao: ate o Ctrl+C)",
    )
    parser.add_argument(
        "--hide-transcript",
        dest="hide_transcript",
        action="store_true",
        help="nao mostra o texto da IA (treino de escuta pura)",
    )
    parser.add_argument(
        "--no-spoke-feedback",
        dest="no_spoke_feedback",
        action="store_true",
        help="nao fala a correcao em voz alta (mostra so o painel)",
    )
    # Phase 9. The dests are distinct from the parent's so that
    # ``tutor --config x stats`` keeps the parent value instead of being
    # overwritten by the subparser's default.
    sub = parser.add_subparsers(dest="command")
    stats_p = sub.add_parser("stats", help="sessoes, erros e evolucao da nota")
    stats_p.add_argument("--config", dest="stats_config", default=None, metavar="PATH")
    stats_p.add_argument("--levels", dest="stats_levels", default=None, metavar="PATH")
    stats_p.add_argument(
        "--limit", dest="stats_limit", type=int, default=10, metavar="N",
        help="quantas linhas cada tabela mostra (padrao: 10)",
    )
    review_p = sub.add_parser("review", help="flashcards das expressoes e erros")
    review_p.add_argument("--config", dest="review_config", default=None, metavar="PATH")
    review_p.add_argument("--levels", dest="review_levels", default=None, metavar="PATH")
    review_p.add_argument(
        "--limit", dest="review_limit", type=int, default=20, metavar="N",
        help="quantos cards revisar (padrao: 20)",
    )
    review_p.add_argument(
        "--answers",
        dest="review_answers",
        default=None,
        metavar="STR",
        help='respostas pre-escritas, ex. "snn" (s=lembrou, n=nao); '
        "sem terminal, sem input",
    )
    return parser


def _print_levels(levels: dict, console: Console) -> None:
    table = Table(title="Niveis")
    table.add_column("Nome")
    table.add_column("CEFR")
    table.add_column("max_tokens", justify="right")
    table.add_column("tts_speed", justify="right")
    table.add_column("end_silence_ms", justify="right")
    table.add_column("ajuda em PT")
    for name, level in levels.items():
        table.add_row(
            name,
            level.cefr,
            str(level.max_tokens),
            f"{level.tts_speed:g}",
            str(level.end_silence_ms),
            "sim" if level.allow_portuguese_help else "nao",
        )
    console.print(table)


def _resolve_scenario(name: str | None) -> str | None:
    """Return the scenario name, validating it against prompts/scenarios/."""
    if not name:
        return None
    candidate = Path(name)
    if candidate.suffix:
        # An explicit path is allowed, but only if it has a stem we know.
        name = candidate.stem
    if name not in available_scenarios():
        return None
    return name


# -- Phase 9: stats ---------------------------------------------------------- #


def _when(iso: str) -> str:
    """``2026-10-01T14:03:11+00:00`` -> ``10-01 14:03`` (local clock, short)."""
    try:
        from datetime import datetime

        moment = datetime.fromisoformat(iso)
        return moment.astimezone().strftime("%m-%d %H:%M")
    except ValueError:
        return iso[:16]


def _score_bar(score: float) -> str:
    filled = max(0, min(10, round(score)))
    return "█" * filled + "░" * (10 - filled)


def cmd_stats(
    cfg_path: str | None, levels_path: str | None, limit: int, console: Console
) -> int:
    cfg, levels = load_all(cfg_path, levels_path)
    db = Database(cfg.storage.db_path)
    try:
        sessions = db.recent_sessions(limit)
        if not sessions:
            console.print("[yellow]nenhuma sessao registrada ainda.[/yellow]")
            return 0

        table = Table(title=f"Ultimas sessoes ({cfg.storage.db_path})")
        table.add_column("id", justify="right")
        table.add_column("data")
        table.add_column("nivel")
        table.add_column("cenario")
        table.add_column("turnos", justify="right")
        table.add_column("nota", justify="right")
        table.add_column("erros", justify="right")
        for row in sessions:
            table.add_row(
                str(row.id),
                _when(row.started_at),
                row.level,
                row.scenario,
                str(row.turns),
                "-" if row.avg_score is None else f"{row.avg_score:.1f}",
                str(row.errors),
            )
        console.print(table)

        counts = db.error_counts(limit)
        if counts:
            errors = Table(title="Erros mais comuns")
            errors.add_column("tipo")
            errors.add_column("ocorrencias", justify="right")
            for item in counts:
                errors.add_row(item.type, str(item.count))
            console.print(errors)

        history = db.score_history(limit)
        if history:
            evolution = Table(title="Evolucao da nota media (mais antiga -> hoje)")
            evolution.add_column("sessao", justify="right")
            evolution.add_column("nota", justify="right")
            evolution.add_column("")
            for session_id, value in history:
                evolution.add_row(
                    str(session_id), f"{value:.1f}", _score_bar(value)
                )
            console.print(evolution)

        order = list(levels)
        current = db.current_level()
        suggestion = db.suggest_level(current or "", order)
        if suggestion:
            window = db.session_scores()
            avg = sum(window) / len(window)
            going_up = order.index(suggestion) > order.index(current)
            direction = "subir para" if going_up else "descer para"
            console.print(
                Panel(
                    f"nota media das ultimas {len(window)} sessoes: {avg:.1f}  "
                    f"(>= {UP_THRESHOLD:g} sobe, <= {DOWN_THRESHOLD:g} desce)\n"
                    f"sugestao: {direction} [bold]{suggestion}[/bold]\n"
                    "[dim]apenas sugestao - nada muda sozinho; use --level[/dim]",
                    title="sugestao de nivel",
                    border_style="yellow",
                )
            )
        return 0
    finally:
        db.close()


# -- Phase 9: review --------------------------------------------------------- #


def _ask(question: str, answers: list[str] | None, console: Console) -> bool | None:
    """True = lembrou, False = nao lembrou, None = sem resposta possivel.

    ``answers`` is the scripted path (``--answers snn``): one character per
    card, consumed in order. None on stdin means there is no terminal to ask,
    and an exhausted script or a Ctrl+C aborts rather than marking the card.
    """
    if answers is not None:
        return answers.pop(0).lower() in ("s", "sim", "y", "yes")
    if not sys.stdin.isatty():
        return None
    try:
        reply = input(question).strip().lower()
    except (EOFError, KeyboardInterrupt):
        console.print()
        return None
    return reply in ("s", "sim", "y", "yes")


def cmd_review(
    cfg_path: str | None,
    levels_path: str | None,
    limit: int,
    answers: str | None,
    console: Console,
) -> int:
    cfg, _levels = load_all(cfg_path, levels_path)
    db = Database(cfg.storage.db_path)
    try:
        fresh = db.new_vocab(limit)
        rest = limit - len(fresh)
        cards = fresh + (db.due_vocab(rest) if rest > 0 else [])
        recurring = db.recurring_errors(limit=limit)
        if not cards and not recurring:
            console.print(
                "[yellow]nada para revisar.[/yellow] "
                f"O deck cresce ao fim de cada sessao com resumo "
                f"({db.vocab_count()} expressoes)."
            )
            return 0

        queue = list(answers) if answers else None
        reviewed = 0
        for position, card in enumerate(cards, start=1):
            # Checked before the card is printed so an exhausted script does
            # not show a card it will not ask about.
            if queue is not None and not queue:
                break
            tag = "nova" if card.is_new else f"caixa {card.box}"
            console.print(
                f"\n[bold]card {position}/{len(cards)}[/bold]  [dim]{tag}[/dim]"
            )
            console.print(f"  [cyan]{card.expression}[/cyan]")
            if queue is None and not sys.stdin.isatty():
                # No terminal: show the deck, ask nothing, touch nothing.
                continue
            reply = _ask("  voce lembrou? [s/N] ", queue, console)
            if reply is None:
                break
            box = db.review_answer(card.id, correct=reply)
            if reply:
                console.print(f"  [green]caixa {box}[/green]")
                reviewed += 1
            else:
                console.print(f"  [red]caixa {box} - volta para o inicio[/red]")

        if recurring:
            console.print(f"\n[bold]erros recorrentes[/bold]  [dim]({len(recurring)})[/dim]")
            for item in recurring:
                console.print(
                    f"  [red]{item.original}[/red]  ->  [green]{item.fix}[/green]  "
                    f"[dim]({item.type}, {item.count}x)[/dim]"
                )
                console.print(f"  [dim]{item.explanation_pt}[/dim]")
        console.print(
            f"\n[dim]{reviewed}/{len(cards)} lembros, "
            f"{db.vocab_count()} expressoes no deck.[/dim]"
        )
        return 0
    finally:
        db.close()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    console = Console()

    if args.command == "stats":
        try:
            return cmd_stats(
                args.stats_config or args.config,
                args.stats_levels or args.levels,
                args.stats_limit,
                console,
            )
        except ConfigError as exc:
            console.print(f"[red]ERRO:[/red] {exc}")
            return 2
    if args.command == "review":
        try:
            return cmd_review(
                args.review_config or args.config,
                args.review_levels or args.levels,
                args.review_limit,
                args.review_answers,
                console,
            )
        except ConfigError as exc:
            console.print(f"[red]ERRO:[/red] {exc}")
            return 2

    try:
        cfg, levels = load_all(args.config, args.levels)
    except ConfigError as exc:
        console.print(f"[red]ERRO:[/red] {exc}")
        return 2

    if args.no_spoke_feedback:
        cfg.pipeline.speak_feedback = False

    if args.list_levels:
        _print_levels(levels, console)
        return 0

    if args.level not in levels:
        available = ", ".join(sorted(levels))
        console.print(
            f"[red]ERRO:[/red] nivel '{args.level}' inexistente. Disponiveis: {available}"
        )
        return 2

    level = levels[args.level]

    scenario = _resolve_scenario(args.scenario)
    if args.scenario and scenario is None:
        console.print(
            f"[red]ERRO:[/red] cenario '{args.scenario}' nao encontrado em "
            f"{SCENARIOS_DIR_LABEL}. Disponiveis: "
            f"{', '.join(available_scenarios())}"
        )
        return 2
    scenario = scenario or DEFAULT_SCENARIO

    try:
        prompt = load_prompts(level=args.level, scenario=scenario)
    except PromptError as exc:
        console.print(f"[red]ERRO:[/red] {exc}")
        return 2

    accent = args.accent or cfg.tts.accent

    console.print("[bold]Configuracao carregada.[/bold]")
    table = Table(title="Resolucao")
    table.add_column("Chave")
    table.add_column("Valor")
    table.add_row("nivel", f"{args.level} ({level.cefr})")
    table.add_row("max_tokens", str(level.max_tokens))
    table.add_row("temperature", f"{level.temperature:g}")
    table.add_row("cenario", scenario)
    table.add_row("frase de abertura", prompt.greeting)
    table.add_row(
        "prompt de sistema",
        f"{len(prompt.system.split())} palavras ~{prompt.estimated_tokens:.0f} tokens "
        f"({len(prompt.sources)} arquivos)",
    )
    table.add_row("acento", accent)
    table.add_row("voz TTS", cfg.tts.voice)
    table.add_row("STT", f"{cfg.stt.model} @ {cfg.stt.device}")
    table.add_row("correcao falada", "sim" if cfg.pipeline.speak_feedback else "nao")
    table.add_row("LLM conversa", f"{cfg.llm_local.model} @ {cfg.llm_local.base_url}")
    if cfg.llm_remote.enabled:
        table.add_row("LLM feedback", f"{cfg.llm_remote.model} (remoto)")
    else:
        table.add_row("LLM feedback", "local (remoto desabilitado)")
    table.add_row("banco", cfg.storage.db_path)
    console.print(table)

    if args.check:
        return 0

    try:
        return asyncio.run(
            run_tutor(
                cfg,
                level_name=args.level,
                level=level,
                scenario=scenario,
                console=console,
                hide_transcript=args.hide_transcript,
                max_turns=args.turns,
            )
        )
    except ConfigError as exc:
        console.print(f"[red]ERRO:[/red] {exc}")
        return 2
    except DeviceError as exc:
        # Environment, not code: the message already lists the devices.
        console.print(f"[red]ERRO:[/red] {exc}")
        return 2
    except Exception as exc:  # noqa: BLE001 - a clear line beats a traceback
        console.print(f"[red]ERRO:[/red] {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
