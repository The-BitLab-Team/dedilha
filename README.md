# Dedilha

Detecta os **acordes** (violão, guitarra, piano), a **linha de baixo** (fundamental de cada acorde e notas do baixo),
os **riffs** (trechos curtos que se repetem, com notas, horários e tablatura)
e, opcionalmente, as **notas cantadas** (G5, F3, A4…) de uma música. A entrada pode ser um link do **YouTube**, um link do **Spotify** ou um **arquivo de áudio**.

> Estado: primeira versão funcional, em texto. Diagramas visuais dos instrumentos ficam para a próxima etapa.
> **Leia a seção [Limitações](#limitações-honestas)** antes de confiar nos resultados.

## Como funciona

```
link YouTube ──► yt-dlp ─┐
link Spotify ──► título/artista ─► busca no YouTube ─► yt-dlp ─┤
arquivo de áudio ───────────────────────────────────────────────┤
                                                                ▼
                                                ffmpeg → WAV 44,1 kHz
                                                                │
                              Demucs (htdemucs_6s, ONNX): voz · guitarra · piano · baixo · outros · bateria
                                                                │
                      ┌─────────────────────────────────────────┼───────────────────┐
                      ▼                                         ▼                   ▼
          basic-pitch (notas polifônicas)               pYIN (altura da voz)   batidas + afinação
                      │                                         │
          acordes por meia batida + Viterbi                notas G5, F3, A4…
                      │
          stem do baixo → linha monofônica + fundamental por acorde
                      │
          stems de guitarra/baixo/piano → notas → trechos que se repetem → riffs (+ tablatura)
```

Detalhes e justificativas: [`docs/PESQUISA.md`](docs/PESQUISA.md).

## Instalar (Windows)

### Opção 1 — um executável só (recomendado)

1. Baixe **`Dedilha-Windows-x64.exe`** na página de **Releases** do repositório (ou o artefato da última execução do GitHub Actions).
2. Dê dois cliques. Na **primeira vez** ele baixa e instala sozinho tudo o que o app precisa
   (Python 3.12 isolado, as bibliotecas, o runtime JavaScript do yt-dlp e o modelo de separação).
   São **~500 MB** e alguns minutos; precisa de internet. Nada é instalado fora da pasta `%LOCALAPPDATA%\Dedilha`
   e não precisa de administrador nem de Python instalado.
3. Nas vezes seguintes ele abre direto a interface no navegador (`http://127.0.0.1:8765`).

O executável não é assinado digitalmente, então o Windows pode mostrar o aviso do **SmartScreen**
("Mais informações" → "Executar assim mesmo") e alguns antivírus podem reclamar de executáveis PyInstaller.
Confira o SHA-256 publicado junto (`Dedilha-Windows-x64.exe.sha256`) se quiser ter certeza de que o arquivo é o mesmo.

Comandos do executável (todos opcionais; sem argumentos ele abre a interface):

```
Dedilha.exe                          # abre a interface web
Dedilha.exe analyze musica.mp3 -i guitar,bass   # qualquer comando da CLI
Dedilha.exe --setup-only             # só instala o que falta e sai
Dedilha.exe --no-model               # não baixa o modelo de separação agora
Dedilha.exe --reinstall              # refaz a instalação do zero
Dedilha.exe --where                  # mostra onde ficam os arquivos
Dedilha.exe --uninstall              # apaga a pasta %LOCALAPPDATA%\Dedilha
```

Se algo falhar, o log da instalação fica em `%LOCALAPPDATA%\Dedilha\logs\setup.log`.

### Opção 2 — a partir do código-fonte

Requisitos: **Python 3.11, 3.12 ou 3.13** ([python.org](https://www.python.org/downloads/); marque "Add to PATH").
Para links do YouTube, instale também o **Node.js** ou o **Deno** (o yt-dlp precisa de um runtime JavaScript).
O ffmpeg **não** precisa ser instalado: vem junto via `imageio-ffmpeg`.

1. Dê dois cliques em **`instalar.bat`** (cria o ambiente `.venv`, instala tudo e baixa o modelo de separação, ~100 MB).
2. Dê dois cliques em **`iniciar.bat`**: abre a interface no navegador (`http://127.0.0.1:8765`).

Em qualquer sistema: `python install.py` (use `--no-model` para pular o download do modelo).

### Gerar o executável você mesmo

```
pip install pyinstaller build
python scripts/build_launcher.py      # gera dist/Dedilha.exe (no Windows)
```

No GitHub, o workflow `.github/workflows/build.yml` roda os testes, compila o `.exe` no Windows, faz um teste de
fumaça (instalação real do zero + análise de um acorde sintético) e publica o executável como artefato.
Ao criar uma tag `v*` (por exemplo `git tag v0.1.0 && git push --tags`), ele também cria uma **Release** com o `.exe`.

## Usar

**Interface web**: cole um link ou arraste um arquivo, escolha os instrumentos, clique em analisar.
Você ganha player com forma de onda, anel do acorde atual, transposição (−/+), abas por instrumento e
botões para copiar o texto ou baixar o JSON. A aba **Riffs** mostra cada riff num cartão (notas, piano-roll,
tablatura, botão "Ouvir riff" e os horários em que ele aparece: clique num horário para ouvir aquele trecho da gravação).

**Linha de comando** (dentro do `.venv`):

```
dedilha doctor                                   # verifica ffmpeg, modelos, runtime JS
dedilha models                                   # baixa o modelo de separação antecipadamente
dedilha analyze "https://www.youtube.com/watch?v=..."
dedilha analyze "https://open.spotify.com/track/..."
dedilha analyze minha_musica.mp3 -i guitar,piano,bass,voice --notes
dedilha analyze minha_musica.mp3 -i bass          # só o baixo
dedilha analyze minha_musica.mp3 -i riffs         # só os riffs
dedilha analyze musica.wav --vocabulary extended --slash --json > saida.json
dedilha serve --port 8765
```

Opções úteis: `--no-separation` (só os acordes da mix, bem mais rápido), `--voice-from-mix` (voz sem separar),
`--vocabulary simple|standard|extended`, `--max-minutes 15`, `--cookies-from-browser chrome`
(quando o YouTube pede login), `--keep-stems pasta` (guarda os stems separados).

### Entradas aceitas

- Links `youtube.com`, `youtu.be`, `music.youtube.com`.
- Links `open.spotify.com/track/...` (o áudio **não** é baixado do Spotify: o app lê título e artista e
  procura a gravação no YouTube; confira se o vídeo escolhido é o certo).
- Arquivos: mp3, wav, flac, m4a, aac, ogg, opus, wma, aiff e vídeos comuns (mp4, mkv, webm…). Limite de 300 MB na interface.

### Saída

Para cada instrumento: lista de acordes com início/fim (`Am  0:12–0:14`), tom estimado, andamento, afinação
(em cents) e avisos. No **baixo**: a fundamental de cada acorde da música (a nota mais tocada pelo baixo naquele
trecho), a extensão (nota mais grave e mais aguda) e a linha de baixo nota a nota (de B0 a G4, uma nota por vez).
Com `--notes`/voz: notas cantadas em notação científica (C4 = dó central, A4 = 440 Hz)
com início e duração.

**Riffs** (`-i riffs`, ligado por padrão): para cada riff, as notas na ordem (`E2 G2 A2 A#2 A2 G2`; acordes
como `E2+B2`), instrumento (guitarra, baixo ou piano), duração e batidas aproximadas, quantas vezes se repete,
**os horários de cada aparição** (com `(+5)` quando a repetição vem transposta 5 semitons acima) e uma sugestão
de **tablatura** para guitarra e baixo em afinação padrão. Um riff é uma sequência de 4 a 24 notas/acordes, com o
mesmo ritmo, que se repete (4 notas precisam de 4 repetições; sequências maiores, de 2). Exige a separação de
instrumentos (Demucs).

## Limitações honestas

- **Download real do YouTube/Spotify e o Demucs com pesos reais não foram executados no ambiente de
  desenvolvimento** (rede restrita). Esses caminhos foram testados com simulações. Rode `dedilha doctor`
  e teste uma música sua primeiro.
- **A precisão foi medida só em música sintética** (~99% dos acordes). Em gravações reais com reverb,
  distorção, bateria e vocais, será menor, e os parâmetros ainda precisam de ajuste com música real.
- Os stems de **guitarra e piano do Demucs são experimentais** (o piano vaza mais). O **baixo** depende do stem
  de baixo do Demucs (sem a separação o app avisa que ele não está disponível); em baixos muito distorcidos ou
  com slap/harmônicos, a linha de notas pode ter erros de oitava. O app marca instrumentos
  quase ausentes e sempre mostra também os acordes da mix. Não distingue violão de guitarra elétrica.
- A separação em CPU é lenta (minutos por música); `--no-separation` pula esta etapa.
- Pela leitura de metadados do Spotify (página pública), o fluxo pode quebrar se o Spotify mudar o HTML.
  Opcional e mais estável: defina `SPOTIFY_CLIENT_ID` e `SPOTIFY_CLIENT_SECRET` (API oficial, gratuita).
- Vozes: o pYIN rastreia uma melodia por vez; coros, harmonias e vozes muito graves/agudas saem piores.
- **Nome:** "Dedilha" foi escolhido por não ter achado produto de música, pacote (PyPI/npm) ou app com esse nome exato, mas isso **não é registro de marca**: antes de vender, consulte o INPI. O nome anterior de trabalho, "Chordify", é um produto comercial existente e não deve ser usado.
- **Termos de uso:** baixar do YouTube pode violar seus termos. Use para estudo pessoal, com músicas que
  você tem direito de usar. O servidor local só escuta em `127.0.0.1` e só aceita links do YouTube e do Spotify.

- **O executável do Windows nunca foi executado pelo autor em Windows**: o instalador foi validado de ponta a ponta
  em Linux e o workflow do GitHub faz o teste de fumaça no Windows. Se a primeira execução falhar, o log em
  `%LOCALAPPDATA%\Dedilha\logs\setup.log` mostra onde.

- **Riffs**: só aparecem repetições **quase idênticas** (mesmas notas relativas e mesmo ritmo, com pequena folga de
  tempo). Se o músico varia o riff, ou se uma nota some na transcrição, aquela repetição não é reconhecida; um
  riff pode também ser encontrado só em parte. Dependem dos stems do Demucs (guitarra/baixo/piano) e a precisão foi
  medida **só em música sintética**. As batidas são aproximadas, e a **tablatura é uma sugestão** (afinação padrão,
  trastes 0–17): a posição real que o músico usou pode ser outra. Sequências longas e muito simples podem render
  coincidências; confira de ouvido com o botão "Ouvir riff" e os horários.

## Testes

```
pip install -e ".[dev]"
pip install --no-deps basic-pitch
pytest              # ~1–2 min; usa música sintética com gabarito, sem rede
```

Os testes cobrem teoria musical, reconhecimento de acordes, notas da voz, linha de baixo, riffs e tablatura, classificação de
links/metadados do Spotify (com respostas simuladas), o pipeline completo (com separador falso), a API web e o
instalador do executável (`launcher/`).

## Estrutura

```
src/dedilha/
  music.py        notas, acordes, tom
  audio.py        ffmpeg, carregar áudio, afinação
  sources.py      YouTube, Spotify, arquivos
  separation.py   Demucs ONNX
  transcribe.py   basic-pitch (ONNX)
  chords.py       acordes: chroma → gabaritos → Viterbi
  vocals.py       notas cantadas (pYIN)
  bass.py         linha de baixo e fundamental por acorde
  riffs.py        riffs: trechos curtos que se repetem (notas → eventos → repetições)
  tab.py          tablatura sugerida (violão/guitarra e baixo)
  pipeline.py     orquestra tudo
  render.py       saída em texto
  cli.py          linha de comando
  web/            API FastAPI + interface (static/index.html)
launcher/         instalador/lançador do executável (só biblioteca padrão)
scripts/          build_launcher.py (PyInstaller), make_smoke_wav.py (teste do CI)
.github/workflows/build.yml   testes + build do .exe + Release
tests/            testes + gerador de música sintética (synth.py)
docs/PESQUISA.md  pesquisa de pacotes e decisões
```

## Próximos passos

1. Medir e ajustar com 5–10 músicas reais.
2. Diagramas de violão (braço) e piano (teclado) com os acordes.
3. Validar acordes com baixo invertido (`--slash`) e os riffs em música real (e afinar os limites do detector).
   Depois: riffs com pequenas variações (repetição aproximada) e outras afinações na tablatura.
4. Voz mais precisa com CREPE/pesto (requer PyTorch/TensorFlow).
5. Assinar o executável (certificado de código) para evitar o SmartScreen.
6. Fluxo opcional para GPU (`onnxruntime-directml`; não testado).
