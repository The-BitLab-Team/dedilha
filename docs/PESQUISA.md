# Dedilha — pesquisa de pacotes e decisões de arquitetura

Objetivo: reaproveitar pacotes prontos (como o ffmpeg) em vez de escrever tudo do zero. Esta é a
pesquisa feita antes de programar e o que foi decidido. Itens marcados com **(não testei)** vêm de
documentação ou de conhecimento prévio, sem verificação neste ambiente.

## 1. Pacotes escolhidos

| Etapa | Pacote | Por que |
|---|---|---|
| Baixar áudio do YouTube | **yt-dlp** (`yt-dlp[default]`) | Padrão de fato, atualizado a cada poucos dias. Aceita também SoundCloud, Bandcamp etc. |
| Converter qualquer formato | **ffmpeg** (do PATH ou `imageio-ffmpeg`) | Decodifica mp3, wav, flac, m4a, ogg, opus, aac, wma, vídeos… O `imageio-ffmpeg` traz o binário dentro do pacote pip: não exige instalar nada no Windows. |
| Separar instrumentos | **demucs-onnx** com o modelo `htdemucs_6s` | O Demucs separa bateria, baixo, voz, outros, **guitarra e piano**. A versão ONNX roda só com numpy + onnxruntime: **sem PyTorch** (que pesaria vários GB). |
| Transcrever notas (polifônico) | **basic-pitch** (Spotify) | Leve, funciona com qualquer instrumento e traz o modelo ONNX dentro do pacote. |
| Rastrear a voz | **librosa.pyin** | Rastreador de altura monofônico, sem modelo para baixar. |
| Áudio / DSP | **librosa**, **soundfile**, **scipy**, **numpy** | Batidas, afinação, reamostragem. |
| Interface | **FastAPI + uvicorn** | Servidor local pequeno; a página é HTML/JS puro, sem build. |

Código próprio (não existia pronto de forma confiável): reconhecimento de acordes a partir das notas,
escolha do vídeo certo no YouTube para links do Spotify, segmentação das notas cantadas, correção de afinação.

## 2. Descobertas que mudaram o projeto

1. **O basic-pitch quebra a instalação em Python 3.12+.** Ele declara dependência de
   `tensorflow<2.15.1`, que não tem versão para esses Pythons. Solução: instalar com `--no-deps`
   (feito pelo `install.py`) e usar o modelo ONNX que já vem no pacote, via onnxruntime.
2. **O Spotify não pode ser baixado** (áudio com DRM). O spotDL e similares leem os metadados e procuram
   a música no YouTube; fazemos o mesmo. O oEmbed oficial do Spotify devolve **só o título, sem o artista**,
   então o artista vem de outras fontes em cascata: API oficial (se `SPOTIFY_CLIENT_ID`/`SECRET` existirem),
   página de embed, meta tags da página da faixa. Isso depende do HTML do Spotify e pode quebrar um dia.
3. **O yt-dlp agora exige um runtime JavaScript (Deno ou Node.js) para o YouTube** desde a versão
   2025.11.12, além do componente `yt-dlp-ejs` (já vem com `yt-dlp[default]`). O `dedilha doctor` avisa se
   faltar.
4. **Os stems de guitarra e piano do Demucs são "experimentais"** segundo o próprio projeto (o de piano
   tem mais vazamento). Por isso o app sempre calcula também os acordes da mix e marca instrumentos quase
   ausentes como indisponíveis. Há um único stem de guitarra: **não distingue violão de guitarra elétrica**.
5. O modelo de separação fica no HuggingFace e é baixado (~100 MB) na primeira vez.

## 3. Alternativas descartadas

| Opção | Motivo |
|---|---|
| **madmom** (acordes) | Sem manutenção recente e com histórico de problemas de compilação em Pythons novos **(não testei)**. |
| **autochord** | Depende de TensorFlow (confirmado na página do pacote): mesmo problema do item 2.1. |
| **chord-extractor** | Depende do plugin nativo Chordino (VAMP), um binário por plataforma **(não testei)**. |
| **Essentia** | Tem detector de acordes, mas a licença AGPL **(do meu conhecimento)** complica um produto comercial. |
| **Demucs original (PyTorch)** | Mais testado, porém exige PyTorch. A versão ONNX evita isso. Pode ser trocada: a interface `Separator` é pequena. |
| **CREPE / torchcrepe / pesto** | Mais precisos que o pYIN na voz, mas exigem PyTorch/TensorFlow. Ficam como melhoria futura. |
| **spotDL** (como dependência) | Baixa e grava arquivos; aqui só precisamos de título/artista e da busca. |

## 4. Como o reconhecimento de acordes funciona

1. O basic-pitch devolve a probabilidade de cada uma das 88 teclas estar soando, em ~86 quadros/s.
2. Dobramos as 88 notas em 12 classes de altura (chroma). Esse "cromagrama aprendido" é mais limpo que o
   de um CQT clássico.
3. Agrupamos por **meia batida** (grade do `librosa.beat`; sem batidas, janelas de 0,4 s).
4. Aplicamos compressão de amplitude (expoente 0,5): uma fundamental dobrada em várias oitavas não
   esmaga as demais notas do acorde.
5. Comparamos com gabaritos de acordes (similaridade de cosseno), com uma penalidade pequena para tipos
   raros e um bônus para a nota do baixo.
6. **Viterbi** suaviza a sequência (acordes mudam pouco) e fundimos trechos curtos.
7. O tom (Krumhansl-Schmuckler) define bemol/sustenido na grafia (Bb, não A#, em Fá maior).
8. Gravações fora de A=440 são corrigidas antes (reamostragem), e os tempos voltam à escala original.

Vocabulários: `simple` (maiores/menores), `standard` (+7, maj7, m7; **padrão**), `extended` (+sus2/4, dim,
aug, m7b5, 5).

## 5. Medidas (todas em música SINTÉTICA)

Sem acesso a gravações reais neste ambiente, a precisão foi medida com músicas sintéticas de gabarito
conhecido (violão, piano, baixo, voz e bateria sintetizados): `tests/synth.py`.

- Tríades C–G–Am–F, mix completa: 99–100% dos instantes corretos; violão e piano isolados: 100%.
- Sétimas (Cmaj7–Am7–Dm7–G7): ~98–100% depois da compressão (sem ela, 58%).
- Tons com bemóis e sustenidos: 97–100%. Gravações +35 e −30 cents: 98–100%, afinação estimada com erro < 8 cents.
- Notas da voz: 46/46 notas de uma melodia teste recuperadas, 0 notas sobrando, com e sem vazamento de outros instrumentos.
- Velocidade da transcrição: ~20× mais rápida que o tempo real (CPU de 2 núcleos).

**Os parâmetros (compressão 0,5, peso do baixo 0,15, estabilidade 0,80) foram ajustados nesses testes
sintéticos.** Música real tem reverb, distorção, vocais dobrados e bateria misturada: espere resultados
piores e plano de ajuste com gravações reais.

## 6. O que não foi testado de ponta a ponta

O ambiente de desenvolvimento só alcançava o PyPI (YouTube, Spotify e HuggingFace bloqueados). Portanto:

- Download real do YouTube e leitura real de links do Spotify: testados só com respostas **simuladas**.
- O `Dedilha.exe` e o instalador no **Windows**: exercitados pelo workflow do GitHub num Windows limpo
  (12/11/2026: instalação do zero, `doctor` e análise de um acorde sintético passaram). Nunca foram abertos por
  uma pessoa num Windows de uso comum (antivírus, SmartScreen, rede doméstica).
- Separação real com o Demucs: no mesmo workflow o modelo foi baixado e separou um acorde sintético de 6 s.
  Isso prova que o caminho funciona, não que a separação é boa: em música real a qualidade não foi medida.
  Nos testes automáticos o fluxo usa um separador falso que devolve os stems verdadeiros.

Primeiro passo no seu computador: `dedilha doctor`, depois um link do YouTube e um arquivo local.

### Baixo

O stem `bass` do Demucs passa pelo mesmo basic-pitch e é limpo para uma linha monofônica (faixa B0–G4,
remoção de fantasmas de oitava/harmônicos mais fracos, notas muito curtas e fragmentos). Além das notas, cada
trecho de acorde da mix recebe a **fundamental do baixo** (classe de altura mais forte no trecho). Em baixo
sintético: 32/32 notas certas com 2 extras. Sem separação, o instrumento aparece como indisponível.

### Riffs

Um *riff* aqui é uma sequência curta (4–24 eventos, um evento = nota ou acorde tocado junto) que se repete
com o mesmo ritmo dentro de um mesmo instrumento. Algoritmo (`src/dedilha/riffs.py`):

1. As notas do basic-pitch de cada stem (guitarra, baixo, piano) viram **eventos** (notas com início a até 90 ms
   são um evento). Para guitarra e baixo, fantasmas de transcrição são descartados antes: harmônicos mais fracos
   12/19/24 semitons acima, notas fracas uma/duas oitavas abaixo de outra mais forte e notas < 30% da força típica.
2. Compara-se cada par de posições por **intervalos em semitons** (logo, o riff transposto também casa) e por
   **intervalos entre inícios** (tolerância `max(70 ms, 15%)`), com uma matriz de comprimentos de repetição
   (a mesma ideia da detecção de estruturas repetidas). Acordes são testados pela nota mais grave (power chords,
   baixo) e pela mais aguda (melodia/piano) quando ≥ 10% dos eventos têm várias notas.
3. Candidatos de comprimento *K* precisam de 4 repetições (K = 4), 3 (K = 5) ou 2 (K ≥ 6), sem sobreposição.
4. **Filtros anti-falso-positivo**: trecho ≤ 10 s, intervalo médio ≤ 0,9 s, sem pausas > 1,5 s dentro, sem
   rajadas (> 25% de intervalos < 90 ms), e rejeita nota repetida/trinado (ritmo uniforme com ≤ 2 alturas ou
   pouca mudança de altura) e acordes cheios (≥ 4 notas) sem movimento melódico, como o ritmo de um violão.
5. Escolha gulosa por cobertura (K × repetições), no máximo 6 riffs por instrumento. O riff mostrado é a primeira
   execução, a menos que outra seja claramente mais "limpa" (notas mais fortes). Transposições entram como
   ocorrências com deslocamento em semitons.
6. Tablatura (`tab.py`): afinação padrão (guitarra E4 B3 G3 D3 A2 E2; baixo G2 D2 A1 E1), trastes ≤ 17, alcance de
   mão ≤ 4 trastes, Viterbi sobre ≤ 12 posições por evento minimizando o deslocamento da mão. Sem tablatura
   se mais de 30% das notas não couberem no braço.

Medidas (sintéticas, `tests/test_riffs.py`): riff de 6 notas num compasso de 2 s achado em 8/8 aparições
(± 0,15 s) pelo basic-pitch real, inclusive a cópia transposta +5; acordes dedilhados, nota repetida, trinado, baixo
sustentado e melodias aleatórias (8 sementes, 10 músicas × 3 min) → 0 riffs falsos; 8 000 eventos em ~0,3 s.
**Não validado em gravações reais**: ruído da transcrição, variações do músico e limites do filtro (tempo, repetições
mínimas) precisam de ajuste com música real. Repetições aproximadas (uma nota diferente) ainda não são agrupadas.

### Como tocar os acordes e capotraste

`src/dedilha/voicings.py` gera, para as 12 fundamentais × 11 tipos de acorde (132 acordes):

- **Violão/guitarra (afinação padrão).** Primeiro uma tabela de formas abertas tradicionais (C `x32010`,
  G `320003`, Am `x02210`, B7 `x21202`…); para o resto, formas móveis: "forma de E" (fundamental na 6ª corda),
  "forma de A" (5ª corda) e, só para sus2, "forma de D" (4ª corda). Escolhe-se a que fica mais perto do começo do
  braço. Um teste confere os 132: só notas do acorde, nenhuma faltando (a quinta pode faltar em acordes de 4
  notas), fundamental no baixo, abertura de no máximo 4 casas e pestana coerente.
- **Piano.** Posição fundamental a partir de C3; com baixo invertido, a nota do baixo entra uma oitava abaixo.
- **Capotraste.** Com o capotraste na casa N, cada acorde é tocado com a forma do acorde N semitons abaixo.
  Mede-se a fração do tempo de acordes que exige pestana (ou forma sem cordas soltas) para N = 0…7; sugere-se o N
  de menor custo (`fração + 0,03 × N`) só se o ganho passar de 0,3. Resultado em progressões comuns:
  Bb–Eb–F–Gm → casa 3 (G–C–D–Em); B–E–F#–G#m → casa 4; C–G–Am–F e F#m–D–A–E → sem sugestão (um único acorde
  com pestana não compensa). Esses dois limites (0,03 e 0,3) são escolha minha, sem estudo com violonistas.

Limites: uma posição por acorde (não a que o músico usou), só afinação padrão, e o critério de dificuldade é
simples (forma aberta contra pestana), sem considerar a troca entre acordes.

A interface busca a tabela uma vez (`/api/voicings`) e desenha os diagramas em SVG; a transposição e o capotraste
só trocam a chave consultada.

### Cancelar uma análise

`POST /api/jobs/{id}/cancel`. O cancelamento é cooperativo: uma análise na fila é cancelada na hora; uma em
andamento para na próxima vez que avisa do progresso. A separação de instrumentos (Demucs) e o rastreio da voz
(pYIN) não avisam durante a execução, então nesses trechos o programa só fica livre quando a etapa termina (pode
levar minutos). Interromper de verdade exigiria rodar a análise em outro processo.

Tempos medidos por etapa num trecho sintético de 77 s (CPU de 2 núcleos, sem o Demucs): voz 12 s (6× mais
rápido que o tempo real), transcrição 1,9 s (40×), batidas 1,3 s, afinação 0,2 s. Depois da separação, a voz é a
etapa mais lenta: uns 40 s numa música de 4 minutos.

## 7. Distribuição: um executável que baixa tudo

Empacotar tudo num único `.exe` ("fat bundle": librosa, numba, onnxruntime, scipy, scikit-learn, yt-dlp…)
seria frágil (imports ocultos, binários nativos) e enorme. Em vez disso, `Dedilha.exe` é um **bootstrapper**
de ~12 MB (só biblioteca padrão, PyInstaller `--onefile`) que carrega dentro de si o *wheel* do Dedilha
e, na primeira execução:

1. baixa o **uv** do GitHub Releases, conferindo o SHA-256 publicado;
2. `uv python install 3.12` (Python gerenciado e isolado, nunca o do sistema);
3. cria um venv e instala o wheel com um `constraints.txt` gerado no build (versões das dependências
   congeladas para Windows/3.12); o `basic-pitch` entra com `--no-deps` (sua dependência TensorFlow quebra no 3.12+);
4. baixa o **Deno** (SHA-256 conferido), runtime JS exigido pelo yt-dlp;
5. baixa o modelo Demucs (~100 MB; falha tolerada: pode ser baixado na primeira análise).

O estado fica em `state.json` (hash do wheel, modelo tentado/ok): reabrir não refaz nada, e uma versão nova do
`.exe` só reinstala o pacote. Tudo vive em `%LOCALAPPDATA%\Dedilha` (`DEDILHA_HOME` muda o local) e sai com
`--uninstall`. O workflow do GitHub Actions compila no `windows-latest`, faz a instalação real do zero e analisa
um acorde sintético antes de publicar o executável.

Riscos conhecidos: executável sem assinatura (SmartScreen), falsos positivos de antivírus em PyInstaller
onefile e dependência de internet na primeira execução.

### Revisão de código (15/11/2026)

Correções feitas depois de uma leitura completa do código:

- **Segurança da interface local.** O campo de link aceitava caminhos de arquivo; como a API devolve o áudio
  analisado, qualquer página capaz de falar com `127.0.0.1` (ou qualquer aparelho da rede, com `--host`) podia
  pedir a leitura de arquivos de mídia do computador. Agora o campo só aceita links, e o servidor confere `Host`
  (contra *DNS rebinding*) e `Origin` (contra pedidos disparados por outros sites).
- **Disco.** Cada análise deixava para sempre o WAV de entrada (até ~150 MB para 15 min), o upload e o MP3 do
  player em `~/.cache/dedilha/jobs`. Agora fica só o MP3 enquanto o programa está aberto; pastas com mais de 3 h
  são apagadas na abertura seguinte.
- **Baixo lento.** A remoção de notas-fantasma comparava todas as notas com todas (8 000 notas: 12 s). Com uma
  janela de tempo: 0,04 s, com resultado idêntico ao anterior em 40 casos aleatórios.
- **Entradas.** Links colados sem `https://` são aceitos; transmissões ao vivo e links curtos `spotify.link`
  recebem uma mensagem clara; domínios que só terminam em "spotify.com" não passam mais por Spotify.
- **Interface.** Arquivos enviados aparecem com o próprio nome (antes: "upload"); recarregar a página retoma a
  análise; uma falha de rede passageira não descarta mais uma análise em andamento.
- **Manutenção.** `basic-pitch` com faixa de versão (`>=0.4,<0.5`), versão do pacote num lugar só, `ruff` no
  workflow, actions do GitHub atualizadas (as antigas rodavam em Node.js 20, já descontinuado).

Não mexi nos parâmetros de detecção (acordes, riffs, voz): sem gravações reais para medir, qualquer ajuste
seria chute. Dois comportamentos conhecidos ficam como estão: uma nota sozinha é rotulada como acorde (ex.: lá
puro vira "Am") e ruído sem música ainda recebe um acorde.

## 8. Próximos passos sugeridos

1. Medir com 5–10 músicas reais que você conhece bem e ajustar os parâmetros.
2. Diagramas: mais de uma posição por acorde, outras afinações, forma certa para baixo invertido.
3. Acordes com baixo invertido (`--slash`, já implementado, ainda não validado em música real).
4. Trocar o pYIN por CREPE/pesto se a voz ficar imprecisa.
5. Detecção de compasso/downbeat para organizar a cifra em compassos.
6. Distinguir violão de guitarra elétrica (hoje é um stem só).
7. Assinar o executável com certificado de código.
8. Riffs com variações (repetição aproximada, alinhamento tolerante a uma nota trocada) e validação em música real.
9. Rodar a análise em outro processo, para o cancelamento valer na hora e a voz ganhar barra de progresso.

## 9. Aviso legal e de nome

Baixar áudio do YouTube pode contrariar os termos de uso do serviço e depende dos direitos de cada
música: use para estudo pessoal.

O nome de trabalho inicial, "Chordify", é um produto comercial existente e foi descartado.
"Dedilha" foi escolhido por não ter produto de música, pacote ou app com o mesmo nome nas checagens
(PyPI, npm, GitHub e buscas na web, incluindo páginas de App Store/Google Play, em 11/11/2026; a consulta direta às lojas e aos registros de domínio não foi possível neste ambiente). Isso **não é registro de marca**:
antes de vender, consulte o INPI.
