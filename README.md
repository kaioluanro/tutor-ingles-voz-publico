# Tutor de Inglês por Voz (local, baixa latência)

Tutor de conversação em inglês que roda 100% local: microfone → VAD → STT → LLM → TTS,
com pipeline em streaming de frases para manter a latência baixa.

Plano completo: [`PLANO_TUTOR_INGLES.md`](PLANO_TUTOR_INGLES.md).

---

## 1. Ambiente de desenvolvimento (medido)

| Item | Valor |
|---|---|
| Sistema | Arch Linux (kernel 6.x), PipeWire/PulseAudio |
| GPU | NVIDIA RTX 4050 Laptop — 6141 MiB |
| Driver | 610.43.02 |
| CUDA toolkit | 13.3 em `/opt/cuda` |
| Python | 3.11.15 (venv em `.venv`) — o sistema tem 3.14, **muito novo** para o Kokoro (`<3.13`) |
| Gerenciador | `uv` 0.11.6 |
| espeak-ng | 1.52 (obrigatório para o Kokoro) |
| Disco | ~97% usado → `models/`, `tools/`, `data/` ficam **fora do git** |

### Instalação do zero (Linux)

```bash
uv venv --python 3.11                       # 3.11 é obrigatório
uv pip install -r <dependências do pyproject.toml>
sudo pacman -S espeak-ng ffmpeg             # espeak-ng é exigido pelo Kokoro
```

### Instalação no Windows (**não testada nesta máquina**)

Esta máquina é Linux/Arch e só isso foi verificado. O caminho abaixo segue a mesma
estrutura e respeita as três dependências frágeis que a §11 do plano aponta
(CUDA 12/cuDNN, espeak-ng no `PATH`, Python <3.13) — mas **nunca foi executado**.

```powershell
# 1. Python 3.11 (o 3.12 também serve; 3.13 não, o Kokoro pede <3.13)
winget install Python.Python.3.11
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1

# 2. dependências do projeto (usa o pyproject.toml)
pip install -e ".[dev]"

# 3. espeak-ng: o Kokoro falha sem ele
winget install espeak-ng.espeak-ng     # depois REABRA o terminal para o PATH valer
espeak-ng --version                   # tem que responder

# 4. CUDA: o faster-whisper precisa de cuDNN 9 + cuBLAS 12. Vem pelo pip, mas no
#    Windows o carregador não acha sozinho -> use o preload do scripts/_common.py
#    (o proprio tutor chama antes de importar faster_whisper).
#    Sem GPU: rode uma vez com config.yaml -> stt.device: cpu / stt.compute_type: int8

# 5. modelo: baixe Qwen3.5-4B-Q4_K_M.gguf e coloque em models/
# 6. llama.cpp: baixe a release com CUDA em https://github.com/ggml-org/llama.cpp/releases
#    e descompacte em tools/llama.cpp/ (llama-server.exe + DLLs)
```

> **Não há `scripts/check_*.py` validado no Windows.** Se algo falhar, comece por
> `python scripts/check_audio.py --record 3` (áudio), depois `check_llm.py` (servidor)
> e `check_stt.py` (CUDA/cuDNN). A tabela §4 e a §5 são as referências de "certo".

---

## 2. Decisões que divergem do plano

| Tema | Plano | Feito | Motivo |
|---|---|---|---|
| Modelo | `Qwen3-4B-Q4_K_M` | **`Qwen3.5-4B-Q4_K_M`** | escolha do usuário; já estava no LM Studio → symlink, economiza disco |
| PyTorch | comando atual do pytorch.org | **`torch==2.14.1+cu126`** | o `ctranslate2` 4.8.2 (faster-whisper) exige **CUDA 12 + cuDNN 9**. O `cu130` instalaria libs cu13 → incompatível |
| llama.cpp | release com CUDA | release **b11306 (CUDA 13.4)** em `tools/llama.cpp/` | resolve contra `/opt/cuda` do sistema; `tools/` é gitignored |
| Flags do servidor | `-ngl 99 -c 8192 --port 8080 --jinja` | + **`-fa on -ctk q8_0 -ctv q8_0 --reasoning off -np 1`** | flash-attn + KV q8_0 libera ~120 MiB; `--reasoning off` evita `reasoning_content` no stream |
| Schema do `turns` | `turns(id, session_id, idx, user_text, ai_text, latency_ms, created_at)` | + **`score INTEGER`** | §6 pede a "evolução da nota média" e a "média das últimas 5 sessões", mas a nota só existe em `Feedback.score`. Sem ela na linha do turno não há o que média |

---

## 3. Ponto de partida rápido

```bash
# terminal 1 — servidor LLM (porta 8080)
./tools/llama.cpp/llama-server \
  -m models/Qwen3.5-4B-Q4_K_M.gguf \
  -ngl 99 -c 8192 --port 8080 --host 127.0.0.1 \
  --jinja --reasoning off -np 1 -fa on -ctk q8_0 -ctv q8_0

# terminal 2 — verificações da Fase 1
.venv/bin/python scripts/check_audio.py --record 3
.venv/bin/python scripts/check_vad.py --seconds 25
.venv/bin/python scripts/check_stt.py --record 5      # ou --file sample.wav
.venv/bin/python scripts/check_tts.py --text "Hello, how are you?"
.venv/bin/python scripts/check_llm.py

# terminal 3 — o tutor (Fase 10; Ctrl+C encerra e gera o resumo)
.venv/bin/python -m tutor --check --level beginner --scenario restaurant  # sem áudio/LLM
.venv/bin/python -m tutor --level beginner --scenario restaurant
.venv/bin/python -m tutor stats
.venv/bin/python -m tutor review
```

**Escolher nível, cenário, sotaque e voz**

```bash
python -m tutor --list-levels                     # nome + CEFR de cada nível
python -m tutor --level intermediate --scenario job_interview
python -m tutor --level advanced --accent british --scenario free_chat
python -m tutor --check                           # só a tabela de resolucao, sem carregar nada
```

| Onde | O que muda |
|---|---|
| `--level` | `prompts/levels/*.md` + `levels.yaml`: `max_tokens`, `temperature`, `tts_speed`, `end_silence_ms` |
| `--scenario` | `prompts/scenarios/*.md`: frase de abertura e o papel do tutor |
| `--accent` | `a` (americana) ou `b` (britânica) — troca a voz do Kokoro no meio da sessão |
| `--hide-transcript` | não mostra a fala da IA (treino de escuta pura) |
| `--no-spoke-feedback` | não **fala** a correção (mostra só o painel); vem de `pipeline.speak_feedback` |
| `--turns N` | encerra sozinho após N turnos; sem a flag, só no Ctrl+C |
| `config.yaml → tts.voice` | `af_heart`, `af_bella`, `am_michael`, `bf_emma`, `bm_george` |
| `config.yaml → audio.*_device` | `null` = padrão do sistema; senão índice/nome do PortAudio |

Tudo que não é comando fica em `config.yaml` (máquina) e `levels.yaml` (pedagógia);
ambos são **estritos**: campo ausente ou digitado errado para com `ConfigError`
apontando arquivo, seção e campo.

Verifique o servidor: `curl -s http://127.0.0.1:8080/health` → `{"status":"ok"}`.

---

## 4. Medições

### 4.1 Latência

| Componente | Métrica | Valor |
|---|---|---|
| **LLM** (Qwen3.5-4B Q4_K_M) | TTFT (tempo até o 1º token) | **63–108 ms** (urllib bruto) / **106–154 ms** (pelo SDK `openai`) |
| | velocidade após o 1º token | **45–54 tok/s** |
| **STT** (faster-whisper `small.en`, cuda/int8_float16) | tempo de transcrição / duração do áudio | **0,06–0,07×** (5,49 s de áudio → 0,38 s) |
| | carga do modelo | **1,4–1,6 s** |
| **TTS** (Kokoro `af_heart`, 24 kHz) | 1º chunk **com** warm-up | **105–137 ms** |
| | 1º chunk **sem** warm-up (chamada fria) | **1083 ms** |
| | RTF (real-time factor) | **0,02–0,03** |
| | carga + warm-up | **5,1–5,9 s** |
| **VAD** (Silero, chunk 512) | latência por chunk | **~32 ms** (tempo de amostra) |

> O Kokoro precisa de um aquecimento: a primeira síntese custa ~1 s, as seguintes ~100 ms.
> O pipeline real deve rodar um warm-up no início da sessão (não por turno).

### 4.2 VRAM — critério de aceite: **< 5,5 GB com os três carregados**

| Processo | VRAM |
|---|---|
| `llama-server` (q8_0 KV + flash-attn) | **2954 MiB** |
| `python` (faster-whisper + Kokoro) | **1042 MiB** |
| **TOTAL** | **3996 MiB** ✅ (limite 5500) |

Comparativo das flags do servidor:

| Config | VRAM do servidor |
|---|---|
| `-fa` off, KV f16 | 3076 MiB |
| `-fa on -ctk q8_0 -ctv q8_0` | **2954–2972 MiB** (−104…−122 MiB) |

Sem perda de desempenho: TTFT 63–108 ms, 45–54 tok/s nos dois casos.

### 4.3 Verificação dos componentes

| Script | Resultado |
|---|---|
| `check_audio.py` | grava e reproduz sem erro ✅ |
| `check_stt.py` | transcreveu **exatamente** `"Hello there. My name is Maria. I would like a table for two people, please."` ✅ |
| `check_tts.py` | frase reproduzida com voz Kokoro (`--warmup` → 1º chunk 105 ms), salva em wav ✅ |
| `check_llm.py` | streaming com TTFT impresso ✅ |
| `check_vad.py` | 25 s de silêncio → **0 eventos** (correto); fala → `start`/`end` corretos ✅ |
| caminho ao vivo completo | mic → VAD → STT transcreveu `"My name is Maria. I would like a table for two people please."` ✅ |

> A partir da Fase 4 os scripts `check_stt.py`/`check_tts.py` rodam sobre as classes reais
> (`tutor.stt.WhisperSTT`, `tutor.tts.KokoroTTS`), então um run verde valida o código que o
> pipeline vai usar. Números novos em §4.6.

### 4.4 Captura e VAD (Fase 3)

Métrica central: **corte do início da frase** = início do turno − início real da frase
(medido por correlação normalizada contra o `.wav` original).

| Cenário | turnos | corr | corte onset (frase 1 / 2) |
|---|---|---|---|
| Áudio direto (sem ruído) | 2 | 1,000 | **+68 ms** / +4 ms |
| Áudio direto + ruído real de sala (SNR 5,2 dB) | 2 | 0,93 | **+68 ms** / +4 ms |
| Loop acústico alto-falante→mic (SNR 7,7 dB) | 2 | 0,48 | +223 ms / −1 ms |
| Loop acústico, ganho 5× (SNR 12,9 dB) | 2 | 0,50 | +1020 ms / −4 ms |
| 25 s de silêncio ambiente | 0 | — | VAD `mean 0,001 / max 0,027 / 0% ≥ 0,5` |
| Fala de 0,397 s (`speech_short.wav`) | 0 (descartada) | — | abaixo de `min_utterance_ms` |

**Conclusão:** o detector preserva o onset (68 ms ≈ 2 chunks de 512 amostras) e não reage a
ruído. O loop acústico **não é proxy válido** — o caminho físico alto-falante→microfone
distorce/atrasa e o corte piora com mais ganho (onset suave do "H" de *Hello*).

### 4.5 Teste ao vivo com voz humana (aceite da Fase 3)

30 s de gravação, usuário diz 3 frases com pausas:

```
START 1,6s → END 4,7s   2168 ms   → Whisper: "My name is Kai"
START 21,9s → END 23,9s  952 ms   → Whisper: "OK."
START 26,1s → END 28,1s 1048 ms   → Whisper: "You fine?"
bruto de 30 s           → Whisper: "My name is Kai Okay You fine?"
```

| Aceite | Resultado |
|---|---|
| falar–pausar–falar = 2 eventos | **3 eventos corretos** ✅ |
| ruído não gera fala | 25 s + 10 s de silêncio → **0 eventos** ✅ |
| início não cortado | conteúdo dos turnos = conteúdo do bruto, nenhuma palavra inicial perdida ✅ |

**Dois achados medidos:**

1. **Transiente de abertura do `InputStream`** — os primeiros ~1,5 s caem de rms **0,139**
   (20× o piso) até 0,0068. *Não* gera falso `START` (testado com 10 s de silêncio puro),
   mas contamina qualquer medição de onset feita nos 2 primeiros segundos — descartá-los.
2. **SNR do microfone ≈ 4–6 dB** — fala rms 0,008–0,012 vs piso 0,0068. O Silero VAD
   detecta por características espectrais e funciona, mas detecção por energia não separa.
   Subir o ganho **não** melhora: o ruído é acústico ambiente, não de quantização.

### 4.6 STT, TTS e playback (Fase 4)

| Métrica | Valor | Meta |
|---|---|---|
| STT: 4,96 s de áudio → texto | **0,39 s** (0,08× real time) | < 1 s |
| STT: carga do `WhisperModel` (uma vez) | 1,39 s | fora do turno |
| TTS: carga do `KPipeline` (uma vez) | 2,75 s | na inicialização |
| TTS: `warmup()` (uma vez) | 0,91 s | na inicialização |
| **TTS: tempo até o 1º chunk** (`--warmup`) | **97 ms** | medir ✅ |
| TTS: RTF total | 0,02 | — |
| Playback: 3 s de áudio sem cortes | `underruns = 0`, duração 3068 ms | sem cortes ✅ |
| **Playback: `stop()`** | **0,025 ms** (silêncio real ≤ 42,7 ms = 1 bloco) | < 200 ms ✅ |
| Meu medidor de fome (produtor lento) | `underruns = 25` | detecta corte ✅ |
| Half-duplex | mutado durante a fala, libera após 300 ms | ✅ |

**Nota sobre streaming do Kokoro:** `en_tokenize` só fatia acima de **510 fonemas**, então
toda frase de tutorial sai como **1 chunk**. Sintetizar frase a frase daria 1º chunk em
51 ms (vs 278 ms) mas adicionaria **~393 ms de silêncio** entre frases — piora a qualidade.
A decisão foi sintetizar o texto recebido como unidade e deixar o streaming frase-a-frase
para o pipeline (Fase 6/7), que é para o que existem `sentence_queue_max` e
`min_words_per_chunk` no `config.yaml`.

### 4.7 Linha de base sequencial (Fase 5)

```
python scripts/bench_latency.py --mode file --turns 5 --wav <5 gravações>
```

O modo `file` empurra as gravações pela **fila de captura**, então elas passam pelo
Silero VAD de verdade (pré-roll, `min_utterance_ms` e a espera de `end_silence_ms`);
só o *callback* do PortAudio é substituído. O modo `live` é idêntico, com a voz vindo
do microfone.

| turno | fala → texto | 1º token | fim LLM | 1ª frase>TTS | 1º áudio | playback | **latência** |
|---|---|---|---|---|---|---|---|
| 1 | +304 | +417 | +652 | +652 | +723 | +741 | **+741** |
| 2 | +95 | +237 | +591 | +591 | +682 | +701 | **+701** |
| 3 | +165 | +300 | +577 | +577 | +651 | +670 | **+670** |
| 4 | +109 | +249 | +595 | +595 | +699 | +723 | **+723** |
| 5 | +78 | +220 | +656 | +656 | +815 | +841 | **+841** |
| **mediana** | +109 | +249 | +595 | +595 | +699 | +723 | **723 ms** |

Média da latência percebida: **735 ms**. Valores em ms a partir do fim da fala.

**Onde vai o tempo (mediana):**

```
fala ─109ms─ texto ─140ms─ 1º token ─346ms─ fim LLM ─104ms─ 1º áudio ─24ms─ som
```

* STT `small.en` em CUDA: **109 ms**
* LLM até o 1º token: **140 ms** (TTFT do llama-server)
* **Geração da resposta inteira: 346 ms** ← o maior bloco, e o único que o
  pipeline em streaming da Fase 6 consegue sobrepor ao TTS/playback
* TTS até o 1º chunk: **104 ms**
* Fila do dispositivo: **24 ms**

**O que essa latência *não* inclui.** Pela definição do plano (§9),
`t_speech_end` já é o instante em que o VAD *declara* o fim da fala — ou seja, a
espera de `end_silence_ms` (1500 ms no nível beginner, 1100 no intermediate,
900 no advanced) **aconteceu antes** e não é contada. Do ponto de vista do
usuário, o tempo total depois que ele para de falar é

```
end_silence_ms + latência  →  beginner: 1500 + 723 ≈ 2,2 s
```

Ambos ficam abaixo da meta de 3,5 s da Fase 6.

**Achado: o microfone não ouve os alto-falantes.** Testado com um tom de 440 Hz —
o monitor do sink grava o tom em amplitude cheia (`rms 0,283`) e o microfone devolve
só o piso de ruído (`rms 0,009`, plano). Ou seja, não há acoplamento acústico
utilizável (DMIC direcional + alto-falantes de teto/base), e **injeção acústica não é
um caminho de benchmark** nesta máquina. `--mode file` existe justamente para isso.

---

### 4.8 Pipeline em streaming (Fase 6)

```
python scripts/bench_latency.py --mode file --pipeline streaming --turns 10 \
    --wav <10 gravações>
```

O mesmo modo `file` da §4.7, agora com `--pipeline streaming`: três tarefas
`asyncio` concorrentes (`producer` → `tts_worker` → `playback`) e o
`SentenceChunker` decidindo quando a primeira frase está pronta para virar áudio.
O script imprime ao final **`playback antes do fim do LLM: 9/10 turnos`**.

| turno | fala → texto | 1º token | 1ª frase>TTS | 1º áudio | playback | fim LLM | **latência** |
|---|---|---|---|---|---|---|---|
| 1 | +290 | +642 | +770 | +887 | +907 | +989 | **+907** |
| 2 | +79 | +220 | +301 | +421 | +457 | +577 | **+457** |
| 3 | +151 | +292 | +366 | +475 | +499 | +673 | **+499** |
| 4 | +95 | +235 | +417 | +541 | +568 | +625 | **+568** |
| 5 | +113 | +252 | +327 | +438 | +467 | +496 | **+467** |
| 6 | +91 | +236 | +439 | +564 | +569 | +630 | **+569** |
| 7 | +79 | +222 | +298 | +412 | +418 | +550 | **+418** |
| 8 | +80 | +226 | +302 | +410 | +453 | +609 | **+453** |
| 9 | +95 | +240 | +424 | +484 | +526 | +424 | **+526** |
| 10 | +120 | +1066 | +1151 | +1265 | +1282 | +1315 | **+1282** |
| **mediana** | +95 | +238 | +392 | +479 | +512 | +617 | **512 ms** |

Média da latência percebida: **615 ms**. A abertura (turno 0, só TTS, sem LLM)
 sai em **+115 ms**.

**Aceite**

| critério | meta | medido |
|---|---|---|
| latência mediana em 10 turnos curtos | ≤ 3,5 s | **512 ms** ✅ |
| `t_playback_start < t_llm_done` | a maioria | **9/10** ✅ |
| pausas audíveis entre frases | não | **pior fronteira 170 ms** ✅ |

Contra a linha de base da §4.7: **723 ms → 512 ms** (−211 ms, −29 %). A diferença
cresce com o tamanho da resposta, porque o baseline espera a resposta *inteira*
(enfileirar a 1ª frase economiza `fim LLM − 1ª frase>TTS = 617 − 392 = 225 ms`
mesmo em respostas de 80 tokens).

**O turno 9 é o único que não começou a tocar antes do fim do LLM.** A resposta foi
`Hello Maria! How can I help you today?` — a primeira frase tem 2 palavras, abaixo
do `pipeline.min_words_per_chunk: 3`, então o chunker a segurou esperando a próxima;
como a última não tem espaço depois do `?`, nada sai até o `flush()` de fim de
fluxo. É o custo documentado de `min_words=3` (Fase 7 pode reavaliar).

**Trim de silêncio (novo em `audio/silence.py`).** Medir sem ele mostra o problema
que o streaming criaria:

```
$ python scripts/check_stream_gaps.py --no-trim
uma chamada  duracao 7.40s  fronteiras 7  silencio mediana 80ms  maximo  490ms
por frase    duracao 9.38s  fronteiras 7  silencio mediana 90ms  maximo 1000ms
pior extra: 890ms  (limite 400ms)          → exit 1
```

Cada chamada ao Kokoro acrescenta ~300 ms de entrada e ~450 ms de saída. Com o trim
(`--verify` confirma que nenhuma palavra some, transcricao identica):

```
$ python scripts/check_stream_gaps.py --verify
uma chamada  duracao 6.58s  fronteiras 7  silencio mediana 80ms  maximo  490ms
por frase    duracao 6.05s  fronteiras 9  silencio mediana 90ms  maximo  170ms
pior extra: 110ms  (limite 400ms)          → exit 0
```

O `tts_worker` agora concatena os chunks de **uma frase inteira**, corta o que está
abaixo do próprio pico (margem de 60 ms em cada lado) e só então entrega ao
playback. Custa ~RTF 0,02 × duração da frase, irrelevante perto dos 90 ms do 1º
chunk. O baseline sequencial continua sem trim, para os números da §4.7
continuarem reproduzíveis.

---

### 4.9 Níveis, cenários e prompts (Fase 7)

```
python -m tutor --level beginner --scenario restaurant
python scripts/bench_latency.py --mode file --pipeline streaming \
    --level advanced --scenario doctor --turns 3 --wav ...
```

`core/prompts.py` monta o system prompt concatenando **`base.md` + `levels/<nível>.md`
+ `scenarios/<cenário>.md`** e extrai do cenário a `Opening line`, que vira o turno 0.
Arquivo ausente ou vazio → `PromptError` nomeando o caminho; cenário sem linha de
abertura → erro apontando o `.md`. Tudo é relido a cada chamada, então editar um
prompt não exige reiniciar.

| combinação | palavras | tokens estimados |
|---|---|---|
| base + nível + cenário (18 pares) | 207–213 | **284–308** (limite 600) |

**Aceite**

| critério | medido |
|---|---|
| conversa em cada nível se distingue | **sim**, 3 níveis ✅ |
| sem markdown/emoji/lista | **0 de 33 respostas** ✅ |

Respostas reais por nível (turno 1, mesma fala do aluno: *"My name IS Kyle."*):

* **beginner / restaurant** — `Nice to meet you, Kyle. What would you like to eat today?`
  (2 frases curtas, presente simples, sem idiom)
* **intermediate / free_chat** — `Nice to meet you, Kyle. I hope your day is going well so
  far. What is one thing you look forward to doing this weekend?`
  (3 frases, pergunta aberta)
* **advanced / doctor** — `Nice to meet you, Kyle. So, what's on your mind? I need to know
  what symptoms you're dealing with and when they started.`
  (em personagem, mais longo)

Para exercitar de verdade o "disagree politely" do avançado foram gerados dois wavs
sintéticos com o próprio Kokoro (`/tmp/adv_*.wav`):

```
aluno: I think social media is destroying our attention spans...
tutor: That's a pretty common take, but have you considered that the algorithm might
       just be training us to scan rather than focus deeply? ...

aluno: Honestly, I believe remote work is a waste of time...
tutor: Honestly, I think you're oversimplifying it; the whole "work from home is a scam"
       narrative feels a bit like a conspiracy theory to me. ...
```

Latência nesses turnos do avançado: **1427 ms** de mediana — acima do beginner
(597 ms) porque `max_tokens` é 220 contra 80, mas ainda 2,4× abaixo da meta de 3,5 s.

**`temperature` por nível.** `levels.yaml` ganhou `temperature` (0.7 nos três, §8) e
`Session.turn` a envia a cada chamada; `llm_local.temperature` continua como padrão de
quem não passa uma (ex.: o worker de feedback da Fase 8).

---

### 4.10 Feedback de correção e resumo (Fase 8)

```
python scripts/bench_latency.py --mode file --pipeline streaming --feedback \
    --turns 5 --wav ... --wav ...
```

`core/feedback.py` tem duas peças: `parse_feedback()` (JSON → pydantic) e o
`FeedbackWorker`, uma fila + uma tarefa de consumidora que só chama o LLM **quando o
sistema está ocioso** (`asyncio.Event` que a `Session` limpa no início de cada turno).

| caminho | comportamento |
|---|---|
| `llm_remote.enabled: true` | correção no modelo remoto → **0 ms** de latência, e no local **só se o remoto falhar e o sistema estiver ocioso** |
| `llm_remote.enabled: false` (padrão medido) | correção local → **+153 ms** de mediana (ver §5.5) |

O rigor por nível sai de `feedback.md`: o arquivo é cortado em
`build_feedback_system(level)`, mantendo só o bloco `### BEGINNER|INTERMEDIATE|ADVANCED`
— 149–155 palavras por nível. O pedido é enfileirado em **`on_llm_done`**, dentro do
pipeline, no instante em que `t_llm_done` é carimbado; a correção fica com o modelo
durante a playback inteira, não no intervalo entre turnos.

**Aceite** (3 corridas por condição, `--mode file --pipeline streaming --turns 5`)

| critério | medido |
|---|---|
| feedback aparece sem atrasar a resposta | **5/5 correções + resumo impressos**, custo **0 ms** no caminho remoto e **+153 ms** no local ✅ / ⚠️ |
| offline (`llm_remote.enabled: false`) continua funcionando | **ok** — é o próprio caso medido ✅ |

```
correcao - turno 1  nota 10/10   ...   correcao - turno 5  nota 5/10
feedback: 5 correcao(oes)  0 descartado(s)  0 pendente(s)

MOST FREQUENT ERRORS: ...
NEW EXPRESSIONS: I am happy to talk with you; Nice to meet you; ...
STRENGTHS: Voce se apresentou claramente ...
FOCUS FOR NEXT TIME: Vamos praticar fazer perguntas sobre o menu ...
```

`build_summary_system()` entrega exatamente essas quatro partes rotuladas; a `Session`
imprime o painel ao fim do `run()` e ele também sai pelo `--feedback` da bancada.

### 4.11 Persistência e progresso (Fase 9)

`storage/db.py` é um único `sqlite3` com `CREATE TABLE IF NOT EXISTS` — sem ORM, sem
migração. O banco é aberto por quem precisa (a `Session` durante a conversa,
`stats`/`review` depois) e fechado em seguida.

```
sessions(id, started_at, ended_at, level, scenario, accent, summary)
turns(id, session_id, idx, user_text, ai_text, latency_ms, score, created_at)
errors(id, turn_id, type, original, fix, explanation_pt)
vocab(id, expression, source_session_id, created_at, last_review, box)   box 1..5
```

Duas decisões que não estão no plano:

- **A linha do turno nasce na transcrição, não no fim.** `on_llm_done` entrega a
  correção **durante a playback** (§4.10), antes de `streaming_turn` retornar. Se a
  linha fosse gravada no fim, a correção chegaria procurando um turno inexistente e o
  `score`/os `errors` se perderiam. A linha abre com `ai_text=""` e `update_turn()`
  fecha com a resposta e a latência.
- **O resumo já veio traduzido uma vez** (`NOVAS EXPRESSÕES:` em vez de
  `NEW EXPRESSIONS:`), e a sessão inteira ficou sem vocabulário, porque
  `parse_new_expressions()` casa o nome em inglês. `prompts/summary.md` agora avisa
  que os quatro rótulos são "program code … copy them verbatim, in English"
  (regressão coberta por `test_summary_system_warns_against_translating_the_labels`).
  Linha ausente devolve `[]`, nunca uma exceção.

**Duas sessões reais** (`bench_latency.py --mode file --pipeline streaming
--turns 2 --feedback --scenario restaurant`):

| sessão | turnos | nota média | erros | expressões |
|---|---|---|---|---|
| 1 | 2 | 10,0 | 0 | 5 |
| 2 | 2 | 8,0 | 1 (`grammar`) | 0 (rótulos traduzidos — antes do fix) |

```
$ python -m tutor stats
Ultimas sessoes (data/tutor.db)     Erros mais comuns
 id │ data        │ nivel      │ turnos │ nota │ erros   tipo    │ ocorrencias
  2 │ 10-01 02:40 │ beginner   │      2 │  8.0 │     1   grammar │           1
  1 │ 10-01 02:40 │ beginner   │      2 │ 10.0 │     0

Evolucao da nota media (mais antiga -> hoje)
 sessao │ nota │
      1 │ 10.0 │ ██████████
      2 │  8.0 │ ████████░░

sugestao de nivel: nota media das ultimas 2 sessoes: 9.0 → subir para intermediate
                   (apenas sugestao - nada muda sozinho; use --level)
```

```
$ python -m tutor review --answers snnss
card 1/5  nova
  Table for one
  caixa 2
...
3/5 lembros, 5 expressoes no deck.
```

`review` é interativo por padrão (`voce lembrou? [s/N]`), e `--answers "snn"` existe
para rodar sem TTY. Leitner simples: acertou sobe uma caixa (teto 5), errou volta à 1.
Os **erros recorrentes** (mesmo `original` 2× ou mais) entram como cards sem caixa,
mostrados depois dos vocabulários. A sugestão de nível olha as **5 últimas sessões com
nota** e nunca muda o nível sozinha.

---

### 4.12 Interface de terminal (Fase 10a)

`python -m tutor` roda a sessão inteira no terminal. `--check` valida a resolução e sai
**sem** carregar áudio/STT/TTS/LLM (útil antes de gastar 5 s de modelo):

```console
$ python -m tutor --check --level beginner --scenario restaurant
Configuracao carregada.
                                   Resolucao
┏━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
│ nivel             │ beginner (A1-A2)                                         │
│ cenario           │ restaurant                                               │
│ frase de abertura │ Good evening! Welcome in. Table for one, or are you ...  │
│ prompt de sistema │ 207 palavras ~290 tokens (3 arquivos)                    │
│ LLM conversa      │ qwen3.5-4b @ http://127.0.0.1:8080/v1                    │
│ LLM feedback      │ local (remoto desabilitado)                              │
│ banco             │ data/tutor.db                                            │
└───────────────────┴──────────────────────────────────────────────────────────┘
```

Sessão real com **uma fala injetada** (`--turns 1`; a fala vai pelo monitor de um
`module-null-sink`, veja §4.5 — o mic real está mudo). O painel do estado é um
`rich.Live` que reescreve a linha atual; as tabelas e caixas são impressas por cima:

```console
abertura
  tutor: Good evening! Welcome in. Table for one, or are you waiting for someone?
                                latencia - abertura
│     0 │       — │       — │       — │    +109 │     +133 │       — │    +133 │
fala encerrada (VAD)
╭─────────────────────── correcao - turno 1  nota 10/10 ───────────────────────╮
│   voce: Hello, my name is Maria.                                             │
│   nenhum erro nesta frase.                                                   │
│   local                                                                      │
╰──────────────────────────────────────────────────────────────────────────────╯

turno 1
  voce: Hello, my name is Maria.
  tutor: Hello Maria! Please sit down. Would you like a menu?
                               latencia - turno 1
│     1 │    +369 │    +711 │    +836 │    +959 │     +966 │    +989 │    +966 │
╭─────────────────────────────────── tutor ────────────────────────────────────╮
│ nivel     beginner · restaurant  ·  american                                 │
│ estado    encerrando...                                                      │
│ latencia  966 ms                                                             │
│ voce      Hello, my name is Maria.                                           │
│ tutor     Hello Maria! Please sit down. Would you like a menu?               │
│ feedback  nota 10/10 · sem erros                                             │
╰──────────────────────────────────────────────────────────────────────────────╯
╭────────────────────────────── resumo da sessao ──────────────────────────────╮
│ MOST FREQUENT ERRORS: ausência do artigo definido antes do substantivo       │
│ NEW EXPRESSIONS: good evening; welcome in; table for one; ...                │
│ STRENGTHS: você foi muito clara e correta ao se apresentar pelo nome         │
│ FOCUS FOR NEXT TIME: adicionar artigos antes dos substantivos na próxima     │
│ frase                                                                        │
╰──────────────────────────────────────────────────────────────────────────────╯
```

**Ctrl+C durante a escuta** (não no meio de um turno) é o caminho normal de saída:
`interrompido pelo usuario.` → painel final → resumo, **uma vez**, e a linha da sessão
fecha com `ended_at` + `summary` no `data/tutor.db`. Numa terceira rodada (latência
**873 ms**) a mesma coisa aconteceu.

| Rodada | turno | latência | resumo impresso |
|---|---|---|---|
| `--turns 1` | 1 | **966 ms** | 1× |
| `--turns 1 --hide-transcript` | 1 | 993 ms | 1× |
| Ctrl+C após 1 turno | 1 | 873 ms | 1× |

`--hide-transcript` troca a fala da IA por `(transcript oculto)` (abertura, turno e
dashboard) e **mantém** a transcrição do aluno — é o modo "escuta pura".

Flags novas: `--check`, `--turns N` (padrão: até o Ctrl+C), `--hide-transcript`.
**270 testes** passam (`tests/test_ui_terminal.py` = 17).

---

### 4.13 Teste de estabilidade (Fase 11) — **falhou, vazamento pendente**

```bash
.venv/bin/python scripts/check_stability.py --minutes 15 --feedback
```

Ele injeta `tests/fixtures/speech_b.wav` no sink virtual (§3) em loop, imprime
`turno / rss / vram / fila / latencia` a cada turno e no fim compara o primeiro com o
último. Sai com código **1** se o RSS crescer > 64 MiB, a VRAM > 128 MiB ou a VRAM
passar de 5632 MiB. **Ele fala em voz alta a sessão inteira** — é playback de verdade,
que é justamente o que está sendo medido.

Resultado de 2026-10-01 (RTX 4050, `--minutes 15 --feedback`):

| Métrica | Início | Fim | Drift | Tolerância | Status |
|---|---|---|---|---|---|
| turnos | — | **181 em 15,0 min** (5,0 s/turno) | — | — | ok |
| RSS | 2517 MiB | 2926 MiB | **+409 MiB** (~2,3 MiB/turno) | ≤ 64 MiB | **falhou** |
| VRAM | 4145 MiB | 4193 MiB | +48 MiB | ≤ 128 MiB e < 5632 | ok |
| fila de captura | 22 blocos | 22 blocos | 0 | — | ok |

O crescimento do RSS é **perfeitamente linear** (não há plateau), mas o **Python está
limpo** — ver §5.9. Sem `--feedback` ainda sobe 1,2 MiB/turno, então a causa está no
núcleo (STT/LLM/TTS/playback), não no feedback nem no rollup.

### 4.14 Contexto do STT e correção falada (Fase 12)

```bash
# confiança da transcrição + veredito do gate
.venv/bin/python scripts/check_stt.py --wav tests/fixtures/speech_b.wav
.venv/bin/python scripts/check_stt.py --wav tests/fixtures/speech_b.wav \
    --prompt "Learner: I would like a table for two people please. Tutor: A table for how many?"
.venv/bin/python scripts/check_stt.py --wav tests/fixtures/speech_b.wav --hotwords "table, people, please"
# pipeline pt-BR do Kokoro (correção falada)
.venv/bin/python scripts/check_tts.py --lang pt-br --warmup --no-play
```

**O gate de confiança não apanha erro confiante.** Mesmo clipe (`speech_b.wav`)
degradado por ruído gaussiano, `model=small.en`:

| Condição | Transcrição | `avg_logprob` | `compression_ratio` | Veredito (−1.0 / 2.4) |
|---|---|---|---|---|
| limpo | `I would like a table for two people, please.` | −0,313 | 0,880 | aceita |
| clipe de 0,4 s (`Hi.`) | `Hi.` | −0,716 | 0,273 | aceita |
| **−5 dB de SNR** | `I look like dangle for two people.` | **−0,647** | 0,818 | **aceita** ← o erro passa |
| −5 dB + `--prompt` | `I would like a table for two people please.` | **−0,294** | 0,896 | aceita |
| −5 dB + `--hotwords` | `I look like a table for two people, please` | **−0,386** | 0,875 | aceita |
| ruído branco puro | *(vazio)* | 0,000 | 0,000 | descartado como sempre |

Conclusão medida: `stt.min_avg_logprob = -1.0` serve para **lixo**; contra
desentendimento quem trabalha é o `initial_prompt`/`hotwords`, que **corrigem a palavra
e ainda sobem a confiança**. `no_speech_prob` ficou **fora** do gate de propósito —
no clipe de 0,4 s ele é 0,112 e derrubaria um turno bom.

**A segunda pipeline do Kokoro não custa VRAM.** Medido com
`torch.cuda.memory_allocated()`:

| | VRAM alocada |
|---|---|
| `KokoroTTS.load()` (inglês, `af_heart`) | 333,1 MB |
| `+ load_feedback()` (`pt-br`, `pf_dora`, mesmo `KModel`) | **+0,0 MB** |
| total | 333,1 MB |

E o warmup em `__aenter__` importa: primeiro chunk do clipe pt-BR **4085 ms** na
primeira execução (download de `pf_dora.pt`, ~1 MB, uma vez só) → **972 ms** sem
warmup → **87 ms** depois do warmup (inglês: 100 ms). RTF pt-BR 0,33 após aquecer.

**End-to-end sem fones** (Whisper real + `llama-server` real + Kokoro real, playback
descartado): 3 turnos com frase errada → **9 blocos de áudio** = 3 respostas +
3 correções em inglês + 3 explicações em pt-BR, com `spoken_pt` vindo do JSON do
modelo. Script em `/tmp/opencode/e2e_spoken.py`. O **teste ao vivo com fone** continua
pendente — não há fone nesta máquina.

---

## 5. Problemas conhecidos e soluções

### 5.1 `libcublas.so.12 is not found` (faster-whisper na GPU)

As libs CUDA 12/9 do `faster-whisper` vêm dos pacotes **pip** `nvidia-*` do venv e **não** estão
no caminho do carregador dinâmico. `LD_LIBRARY_PATH` não basta de forma confiável → usar
**precarregamento com `ctypes.RTLD_GLOBAL`** antes de `import faster_whisper`:

```python
# scripts/_common.py → preload_cuda_libs()
# precarrega libcudnn_*, libcublasLt, libcublas do .venv/lib/python3.11/site-packages/nvidia/
```

Todo script (e o futuro `src/tutor/stt/whisper_stt.py`) **deve chamar `preload_cuda_libs()`
antes** de importar `faster_whisper`.

Fallback, se falhar: `device: cpu` + `compute_type: int8` em `config.yaml`.

### 5.2 Microfone clipando (`peak > 1.0`)

Causa real: um vídeo do YouTube reproduzia pelos alto-falantes e o microfone captava.

```
com Firefox tocando : peak=1.203  rms=0.120  VAD p(média)=0.578  57,8% das amostras ≥ 0.5  ← ruído "vira fala"
silêncio real        : peak=0.236  rms=0.022  VAD p(máx)=0.027   0% ≥ 0.5                  ← piso correto
```

Ações: `pactl set-sink-input-mute <id> 1` (mudo a aplicação) e
`pactl set-source-volume @DEFAULT_SOURCE@ 60%`.

> **Fase 3:** calibrar `audio.vad_threshold` medindo o piso de ruído do ambiente real do
> usuário (script pronto em `/tmp/vad_noise.py` — mede 6 s sem fala e imprime a distribuição
> de probabilidades). Se o piso passar de ~0.3, subir o threshold.

### 5.3 `sd.rec()` bloqueante morre quando `sd.play()` inicia

**Achado definitivo da Fase 3:** mesmo aguardando corretamente o `sd.wait()` do recording,
`sd.rec()` + `sd.play()` concorrentes devolve **zeros** a partir do instante em que o
playback começa (transiente inicial presente, resto zerado):

```
rec inicia 1,0 s  →  rms 0,044 / 0,015 / 0,002
sd.play() inicia  →  rms 0,00000 exato até o fim  ← recording morto
gravacao pura (sem play): rms 0,0237, 0 zeros     ← microfone ok
```

Não é o microfone nem o driver: é a API bloqueante do `sounddevice`.

**Solução usada no pipeline:** `AudioCapture` mantém um `sounddevice.InputStream` com
*callback* aberto enquanto `sd.play()` roda — testado, **dados finitos, `bad_cb=0`**.

Regra: **nunca misturar `sd.rec()` com `sd.play()`**; usar sempre stream com callback
(`AudioCapture`/`Playback`).

### 5.4 Gotchas das bibliotecas

- **Silero VAD:** `VADIterator(model, threshold, sampling_rate, min_silence_duration_ms, speech_pad_ms)`
  recebe a taxa **no construtor**; a chamada `vad(x)` é `(x, return_seconds=False, time_resolution=1)` —
  passar a taxa como 2º argumento **não** funciona (retorna segundos).
- **`tts_speed`** é **por nível**, em `levels.yaml` (0,88 / 1,0 / 1,05), não em `config.yaml`.
- **llama.cpp:** flags confirmadas com `--help` (`-fa`, `-ctk/-ctv`, `--jinja`, `--reasoning`, `-np`).
  O servidor aceita `extra_body` (`cache_prompt`, `chat_template_kwargs.enable_thinking`).
- **`pkill -f`/`pgrep -f`** com um padrão que aparece no cmdline do próprio shell **mata a shell**
  em execução. Usar `ps -eo pid,args | grep -E "llama-serve[r]"`.
- **`torchaudio` não pode ficar instalado** ao lado de `torch 2.14.1+cu126`: a versão que o
  uv resolveu (2.11.0, CUDA 13.0) quebra `import kokoro` com erro de ABI
  (`torch 12.6/13.0 mismatch`). Nada no stack o usa — `kokoro` não importa, `silero-vad`
  só no *extra* `audio` opcional e `transformers` trata como opcional. Removido do
  `pyproject.toml`. Se aparecer de novo: `uv pip uninstall torchaudio`.
- **Kokoro fatia em ≤510 fonemas** (`en_tokenize`), então frases curtas viram **1 chunk só**
  e "streaming" não acontece. Fragmentar por frase dá 1º chunk mais rápido mas custa
  ~393 ms de silêncio por fronteira — ver §4.6.

### 5.5 Feedback local custa ~153 ms de latência (KV cache de um único slot)

**Sintoma.** Com `llm_remote.enabled: false` a mediana de latência sobe de **527 → 680 ms**
(3 corridas por condição). Não é fila: as correções terminavam **3 s antes** do turno
seguinte e ainda assim cada turno pagava ~105 ms entre `t_stt_done` e o primeiro token.

**Causa medida.** O `llama-server` roda com `-np 1` — **um slot, um KV cache**. O turno da
conversa e a correção usam dois system prompts que não têm nada em comum, então a cada
alternância o prefixo comum é ~0 e o servidor refaz o prefill inteiro:

```
system prompt do feedback = da conversa (prefixo identico)  ->  mediana 527 ms  = baseline
system prompt do feedback = o proprio (dois em nada iguais) ->  mediana 627 ms
```

**Cancelar não resolve.** `llama-server` termina a geração abandonada mesmo com o HTTP
derrubado — logo depois de um `cancel()`, a requisição seguinte ficou **+595 ms** mais
lenta. Por isso `FeedbackWorker` não tem `interrupt()`: só perderia o resultado.

**O que foi feito**
1. Enfileirar a correção em `on_llm_done` (dentro do pipeline), não no fim do turno →
   a correção aproveita a playback inteira. Levou **1169 → 680 ms** e as correções
   entregues de 1/5 para **5/5**.
2. Aceitar o residual **+153 ms** no caminho local e documentá-lo. Quem não quer esse
   custo liga `llm_remote.enabled: true` (o plano já prefere o remoto para o
   feedback justamente por isso).

**Não testado a fundo (deixado registrado):** `-np 2` (2 slots × 4096 = mesmo KV total,
`kv_unified=false`, só +52 MiB de VRAM) derrubou o custo aparente para ~+50 ms, mas o
baseline também se moveu e faltaram amostras para concluir. Também não se testou
unificar os prompts (mandar as instruções de JSON na *mensagem do usuário* e manter o
system prompt idêntico) — zero mudança de infra, mas coloca uma regra condicional
dentro do prompt do tutor.

### 5.6 `espeak-ng` ausente (Kokoro)

O Kokoro fala via `espeak-ng`; sem ele a síntese falha na primeira frase e a sessão
abre com a abertura muda. Verificação:

```bash
espeak-ng --version          # tem que responder
.venv/bin/python scripts/check_tts.py --text "Hello, how are you?"
```

- **Linux:** `sudo pacman -S espeak-ng` (Debian/Ubuntu: `sudo apt install espeak-ng`).
- **Windows:** `winget install espeak-ng.espeak-ng` e **abra um terminal novo** — o
  `PATH` só é reaproveitado nas próximas janelas. Se o erro continuar, ponha a pasta
  `bin/` do espeak-ng no `PATH` do usuário.
- Depois de instalar, rode o `check_tts.py` **antes** de abrir o tutor: é 1 s de teste
  contra 10 s de modelo carregando.

### 5.7 Servidor LLM fora do ar

`python -m tutor` **não morre mais** por isso (Fase 11). O que acontece agora:

1. o turno falha com `LLM indisponivel (http://127.0.0.1:8080/v1): sem conexao - reconectando...`;
2. o tutor espera no `GET /health` até `pipeline.reconnect_timeout_s` (20 s);
3. voltou → **repete o turno uma vez** (timeline nova, playback limpo);
4. não voltou → `LLM ainda fora do ar apos 20s - turno descartado.` e a conversa segue.

O gate de feedback fica fechado durante a espera, senão a correção de fundo roubaria
o único slot (`-np 1`) justamente quando estamos tentando reconectar.

Diagnóstico rápido:

```bash
curl -s http://127.0.0.1:8080/health     # {"status":"ok"}
ps -eo pid,args | grep -E "llama-serve[r]"
```

> O `--health` devolve **200 mesmo com o servidor gerando** (verificado); 503 aparece
> só enquanto o modelo ainda carrega. Por isso dá para usá-lo como sinal de "posso
> falar agora?" sem interromper nada.

### 5.8 Microfone ausente ou dispositivo errado

`audio.input_device: null` usa o padrão do sistema. Se o padrão for errado (ou não
houver), o tutor aborta com a lista em vez de um traceback:

```
ERRO: sem microfone: ...
  input: 9: pipewire, 10: default, 11: Ryzen HD Audio Controller Digital Microphone ...
  Confira `audio.input_device` / `audio.output_device` no config.yaml (null = padrao do sistema).
```

Liste os dispositivos e ponha o índice ou o nome no `config.yaml`:

```bash
.venv/bin/python -c "import sounddevice as sd; [print(i, d['name']) for i, d in enumerate(sd.query_devices()) if d['max_input_channels']]"
```

Em laptop com DMIC o piso de ruído é alto: meça com `scripts/check_vad.py --seconds 25`
e suba `audio.vad_threshold` se o silêncio passar de ~0.3 (§5.2).

---

### 5.9 RSS cresce ~1,2 MiB por turno (vazamento de nível C — **em aberto**)

O soak de 15 min (§4.13) falha por RSS, não por VRAM. Diagnóstico já feito:

| Ferramenta | Resultado | Conclusão |
|---|---|---|
| `tracemalloc.start(15)`, diff turno 4 → 16 | **vazia** | nada em Python |
| `sys.getallocatedblocks()` | ~2 928 000, plano | nada em Python |
| `len(gc.get_objects())` | ~532 150, plano | nada em Python |
| `gc.collect()` + `malloc_trim(0)` | devolve só **6 MiB** | não é fragmentação do heap principal |
| `check_stability.py --minutes 3.5` **sem** `--feedback` | +86 MiB em 63 turnos | **não** é feedback/rollup |
| `session.capture.queue.qsize()` por turno | constante em **22** | não é a fila de captura |

Portanto a memória fica retida em **C** (bibliotecas: numpy/torch/ctranslate2/
PortAudio), e não em objetos Python. Pendência da Fase 11.

Cuidado ao reproduzir: `/proc/self/statm` é
`size resident shared text lib data dt` — use **`[1]`** (resident). Medir `[2]`
devolve *shared* (~620 MiB) e parece "estável", escondendo o vazamento.

Para checar rapidamente se a correção funcionou, use uma corrida curta:

```bash
.venv/bin/python scripts/check_stability.py --minutes 3.5        # sem feedback
.venv/bin/python scripts/check_stability.py --minutes 3.5 --feedback
```

### 5.10 O Whisper ouve a palavra errada e a conversa sai do trilho (Fase 12)

**Porta de confiança.** Se `avg_logprob < stt.min_avg_logprob` ou
`compression_ratio > stt.max_compression_ratio`, o turno é **descartado antes** do LLM
e do histórico — você vê `transcricao de baixa confianca (logprob -1.43, ratio 2.61)
- pedindo repeticao.` e ouve a frase pronta de `prompts/clarify.md`. Sem isso, um
chute vira frase no contexto e arrasta todos os turnos seguintes.

O limite veio de medição, não de chute (§4.14). Ver o veredito de um wav:

```bash
.venv/bin/python scripts/check_stt.py --wav gravacao.wav
```

**Pista de contexto.** `stt.use_dialogue_context: true` manda as últimas ~4 mensagens
como `initial_prompt`, e `stt.hotwords` dá palavras do cenário — é o que corrige
`fable` → `table` num clipe ruidoso (§4.14). Para desligar: `stt.use_dialogue_context: false`.

**Correção falada.** `pipeline.speak_feedback: true` (padrão) fala a frase corrigida em
inglês e a explicação em português **só quando houver erro**, esperando até
`pipeline.speak_feedback_wait_s` (4 s) pela correção em segundo plano. A linha em
português vem do JSON do modelo (`spoken_pt`), com fallback para
`errors[0].explanation_pt`. Desligue com `--no-spoke-feedback` ou
`pipeline.speak_feedback: false`.

### 5.11 A voz em português (`pf_dora`) não baixa

O `pf_dora.pt` (~1 MB) vem do Hub do Hugging Face **na primeira síntese**. O warmup de
`__aenter__` pega isso no início da sessão; sem rede você vê:

```
[AVISO] correcao falada desativada: <erro do Hub>
```

A sessão segue, apenas sem a parte falada em português (`--no-spoke-feedback` produz o
mesmo efeito, de propósito). Teste isolado:

```bash
.venv/bin/python scripts/check_tts.py --lang pt-br --warmup --no-play
```

---

## 6. Estrutura

```
config.yaml          configuração (audio/stt/llm/tts/pipeline/llama_server)
levels.yaml          configuração por nível (max_tokens, end_silence_ms, tts_speed)
pyproject.toml       dependências (Python >=3.11,<3.13) e pytest
scripts/             verificações manuais (check_*.py) + _common.py
src/tutor/
  config.py          modelos pydantic + load_config/load_levels (erros nomeando arquivo/campo)
  errors.py          DeviceError + lista de dispositivos (Fase 11: microfone/saida)
  llm/base.py        protocolo LLMClient (stream / complete)
  llm/errors.py      is_llm_unavailable/describe: 4xx nao e "servidor fora do ar" (Fase 11)
  llm/health.py      GET /health + wait_for_server (reconexao, Fase 11)
  llm/openai_compat.py  cliente AsyncOpenAI; filtra campos só-locais do remoto
  llm/router.py      conversa = sempre local; feedback = remoto se habilitado
  __main__.py        CLI: --level --scenario --config --accent --list-levels
                     + --check --turns --hide-transcript (Fase 10)
                     + --no-spoke-feedback (Fase 12)
                     + subcomandos stats e review (Fase 9, so tocam no sqlite)
  audio/capture.py   InputStream com callback (nunca sd.rec); pre-roll do TurnDetector
  audio/vad.py       TurnDetector: Silero VAD + máquina de estados + mute half-duplex
  audio/playback.py  OutputStream 24 kHz com fila; end_of_stream, stop, wait_until_done
  audio/silence.py   trim_silence: corta o padding do Kokoro entre frases
  stt/whisper_stt.py faster-whisper; preload de cuDNN/cuBLAS; transcribe em 1 worker;
                     devolve Transcript (avg_logprob/compression_ratio), Fase 12
  tts/kokoro_tts.py  Kokoro; clean_for_tts; synthesize_stream em thread + asyncio.Queue;
                     duas pipelines (en + pt-br) sobre o mesmo KModel, Fase 12
  timeline.py        marcas do plano §9 + tabelas/rich de latência
  core/chunker.py    SentenceChunker: frases prontas a partir de tokens
  core/dialogue.py   system (+ rollup) + saudação + histórico (pares) + fala atual;
                     record() devolve os pares ejetados (Fase 11)
  core/prompts.py    base.md + levels/ + scenarios/ -> system prompt + opening line
                     (+ build_feedback_system / build_summary_system, Fase 8;
                       parse_new_expressions -> deck do review, Fase 9;
                       build_rollup_system -> prompts/rollup.md, Fase 11;
                       load_clarify -> prompts/clarify.md, Fase 12)
  core/pipeline.py   sequential_turn (baseline) e streaming_turn (3 tarefas);
                     on_llm_done devolve o modelo ao worker de feedback;
                     on_state dispara o estado "speaking" (Fase 10)
  core/session.py    Session: devices, modelos, listen/greet/turn/run/report,
                     gate de ociosidade; renderiza tudo via TerminalUI (Fase 10),
                     grava sessao/turnos/resumo no sqlite (Fase 9) e faz
                     reconexao do LLM + rollup do historico (Fase 11), gate de
                     confianca + correcao falada (Fase 12)
  core/feedback.py   parse_feedback (JSON -> pydantic) + FeedbackWorker
                     (fila, gate, summarize e rollup do historico);
                     Feedback.spoken_correction_pt (Fase 12)
  storage/db.py      sqlite3: sessions/turns/errors/vocab, Leitner, stats, review
  ui/__init__.py     exporta TerminalUI, converse, run_tutor
  ui/terminal.py     rich: Live de estado, dashboard final, caixa de correcao
                     (Fase 10a) + converse()/run_tutor() fora do argparse
tests/               pytest (lógica pura + fixtures .wav) — 338 testes
prompts/             base.md, levels/{beginner,intermediate,advanced}.md,
                     scenarios/{free_chat,restaurant,job_interview,travel,doctor,small_talk}.md,
                     feedback.md (JSON + rigor por nivel), summary.md (resumo final),
                     rollup.md (digest do historico ejetado, Fase 11),
                     clarify.md (linhas EN:/PT: quando o STT nao confia, Fase 12)
models/ tools/ data/ fora do git (binário e downloads)
scripts/bench_latency.py  mede N turnos (--pipeline sequential|streaming, --feedback)
scripts/check_stream_gaps.py  mede o silêncio entre frases geradas por peça
scripts/check_stability.py  sessao longa: RSS/VRAM por turno (Fase 11)
```

`.gitignore` cobre `models/`, `tools/`, `data/`, `.venv/`, `*.wav`.

### Erros de configuração

`config.yaml` e `levels.yaml` são **estritos**: todos os campos são obrigatórios e campos
desconhecidos são rejeitados. Qualquer problema vira `ConfigError` nomeando o arquivo, a
seção e o campo — nunca um traceback do pydantic:

```
$ python -m tutor --config /tmp/quebrado.yaml
ERRO: quebrado.yaml: configuracao invalida
  - audio.input_device: Field required
  - audio.output_device: Field required
  - audio.capture_sample_rate: Field required
  - audio.vad_chunk: Field required
  - audio.vad_threshold: Field required
  - audio.min_utterance_ms: Field required
  - audio.max_utterance_s: Field required
  - audio.pre_roll_ms: Field required
  - audio.post_playback_mute_ms: Field required
```

---

## 7. Checklist de progresso

Ver [`PLANO_TUTOR_INGLES.md` §13](PLANO_TUTOR_INGLES.md).
