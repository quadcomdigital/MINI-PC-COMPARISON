# Ranking modelli 32B/70B e fine-tuning LoRA/QLoRA

**Edizione:** analisi best-effort con fonti verificabili consultate durante la redazione.  
**Priorità:** qualità standard dei 32B, poi hardware realistico e fine-tuning PEFT.  
**Avvertenza temporale:** non uso punteggi di modelli/versioni future non verificabili in una fonte primaria. Le classifiche sono quindi ancorate ai checkpoint e ai report ufficiali citati, principalmente fino a maggio 2025; un benchmark pubblicato successivamente va aggiunto solo se mantiene stesso dataset, split, prompt, numero di shot e modalità di decoding.

## 1. Metodologia

### 1.1 Perimetro

Sono considerati modelli testuali densi intorno a 32B e 70/72B, non modelli vision, MoE da 30B-A3B o 235B-A22B, salvo menzione come controllo. Il perimetro principale è:

- **Qwen3-32B**, denso, modalità `thinking` e `non-thinking`; checkpoint Apache 2.0, 32,8B parametri, 128K con YaRN [1][2].
- **DeepSeek-R1-Distill-Qwen-32B**, denso, distillato da R1 sulla base Qwen2.5-32B; 32B, MIT nel repository del distillato [3][4].
- **Qwen2.5-32B-Instruct**, denso, modello general-purpose instruction-tuned, 32,5B; Apache 2.0 [5].
- **Qwen2.5-Coder-32B-Instruct**, denso specializzato per codice, 32,5B; valutato separatamente perché non è corretto confrontare un coding specialist con un generalista come se avessero lo stesso obiettivo [6].
- **QwQ-32B/Preview**, reasoning baseline storico, incluso solo come riferimento perché Qwen3 dichiara di superarlo su 17/23 benchmark [1].
- Riferimenti 70/72B: **Qwen2.5-72B-Instruct**, **Llama 3.3 70B-Instruct** e **DeepSeek-R1-Distill-Llama-70B** [5][7][8]. Mistral Large 2 è 123B, quindi fuori dalla classe 70B e viene citato solo come controllo di scala [9].

### 1.2 Benchmark usati

- **MMLU:** conoscenza multitask; il report Qwen usa 5-shot.
- **MMLU-Pro:** versione più difficile e più robusta del precedente MMLU; il report Qwen3 usa 5-shot + CoT per i base model.
- **MMLU-Redux:** revisione di MMLU, usata nel post-training Qwen3.
- **GPQA/GPQA-Diamond:** domande scientifiche di livello graduate; non confondere GPQA generico con GPQA-Diamond.
- **GSM8K:** aritmetica scolastica; 4-shot + CoT nel confronto Qwen3 base.
- **MATH/MATH-500/AIME:** matematica, con AIME molto più piccolo e quindi più sensibile a sampling e contaminazione.
- **HumanEval, MBPP, EvalPlus, MultiPL-E, LiveCodeBench:** codice; `EvalPlus` è una media di HumanEval, MBPP e versioni con test ampliati, mentre LiveCodeBench cambia nel tempo.
- **IFEval:** instruction following; riporto la variante dichiarata dalla fonte, ad esempio `strict-prompt`.
- **Arena-Hard, MT-Bench, LiveBench:** utili per allineamento/preferenza o valutazione dinamica, ma non sono equivalenti a un test di conoscenza standard.

Il report Qwen3 elenca esplicitamente i protocolli: MMLU 5-shot, MMLU-Pro 5-shot CoT, GPQA 5-shot CoT, GSM8K 4-shot CoT, MATH 4-shot CoT, coding 0-shot e MGSM 8-shot [1]. Per il post-training Qwen3, GPQA-Diamond usa 10 campioni per domanda, AIME 64 campioni e temperatura 0,6 [1][2]. DeepSeek usa generazione massima 32.768 token, temperatura 0,6, top-p 0,95 e 64 risposte dove richiesto [4].

**Date di riferimento dei punteggi:** report Qwen3 14 maggio 2025 [1]; model card Qwen3-32B-AWQ 14 maggio 2025 [2]; DeepSeek-R1 22 gennaio 2025 [4]; Qwen2.5-LLM 19 settembre 2024 [5]; Qwen2.5-Coder 12 novembre 2024 [6]; Llama 3.3 6 dicembre 2024 [7][10]; pagina speed benchmark H20 consultata a settembre 2026 [11].

### 1.3 Regola di comparabilità

Non esiste una singola media scientificamente valida che ordini insieme:

1. base model e instruction model;
2. ragionamento con migliaia di token di CoT e risposta non-thinking;
3. HumanEval e MMLU-Pro;
4. punteggi vendor su prompt proprietari e una replica indipendente;
5. `GPQA` e `GPQA-Diamond`.

Perciò il ranking numerato è **a profili**: general-purpose, reasoning/matematica, coding e instruction following. Una posizione più bassa in un profilo non significa modello peggiore in assoluto: significa che la fonte pubblica non consente di dimostrare il primato su quel sottoinsieme.

### 1.4 Legenda delle evidenze

- **[S]** fonte primaria: paper, model card o specifica del produttore.
- **[C]** numero dichiarato dal vendor/produttore; attendibile come risultato ufficiale, non come verifica indipendente.
- **[M]** misura riproducibile di una fonte terza o benchmark runtime; ambiente e configurazione sono parte del dato.
- **[P]** proxy: confronto su benchmark vicino, ma non identico o non omogeneo.
- **[E]** stima ingegneristica; non va presentata come misura del modello.
- **[€]** costo/prezzo; è uno snapshot e non una garanzia di disponibilità.
- **[Δ]** differenza calcolata in questo documento a partire da numeri citati.

Tutti i punteggi sono percentuali, salvo rating, MT-Bench/AlignBench o quando indicato diversamente.

---

## 2. Ranking qualità — classe 32B (priorità)

### 2.1 Ranking general-purpose più difendibile

**Classifica primaria per chi vuole un solo modello 32B di qualità, con ragionamento, codice, multilingue e possibilità di LoRA:**

1. **Qwen3-32B, thinking** — miglior candidato general-purpose 32B documentato nel perimetro: `MMLU-Redux 90,9`, `GPQA-Diamond 68,4`, `LiveBench 74,9`, `AIME24 81,4` [S][C][2]. Il checkpoint supporta anche `non-thinking`, quindi la stessa base può fare risposte rapide e reasoning a budget variabile.
2. **DeepSeek-R1-Distill-Qwen-32B** — secondo come modello operativo general-purpose, ma **primo per reasoning matematico in questa classe**: `AIME24 72,6`, `MATH-500 94,3`, `GPQA-Diamond 62,1`, `LiveCodeBench 57,2`, Codeforces `1691` [S][4]. È molto forte sulle risposte difficili, ma il CoT lungo riduce throughput e può essere meno conveniente per dialogo rapido.
3. **Qwen2.5-Coder-32B-Instruct** — **primo per coding specialist** sui numeri ufficiali disponibili: HumanEval `92,7`, MBPP `90,2`, LiveCodeBench `31,4`, Aider `73,7`, Spider `85,1`, CodeArena `68,9` [C][6]. Non è la scelta primaria per un assistente generalista se il dataset non è soprattutto codice.
4. **Qwen2.5-32B-Instruct** — ancora ottimo generalista e probabilmente il più semplice da adattare con SFT/LoRA: MMLU-Pro `69,0`, GPQA `49,5`, MATH `83,1`, GSM8K `95,9`, HumanEval `88,4`, MBPP `84,0`, MultiPL-E `75,4`, IFEval `79,5`, Arena-Hard `74,5`, MT-Bench `9,20` [C][5]. È una baseline forte, ma Qwen3-32B thinking è superiore nei benchmark di reasoning pubblicati più recenti.
5. **QwQ-32B** — baseline reasoning precedente. Qwen3 dichiara che Qwen3-32B thinking lo supera in 17/23 benchmark [C][1]; i numeri di repliche MMLU-Pro non ufficiali possono divergere, quindi non li uso per una classifica assoluta.

**Perché Qwen3 è al primo posto:** non perché esista una media universale, ma perché è l'unico dei candidati 32B qui esaminati con una combinazione documentata di reasoning alto, modalità rapida, benchmark dinamico e quantizzazione ufficiale AWQ. La scelta può cambiare se il carico è quasi tutto codice o matematica competitiva.

### 2.2 Confronto standard dei base model

Il confronto più pulito fra i modelli densi è la **Tabella 4 del report tecnico Qwen3**, che usa la stessa pipeline sui base model [S][1]:

| Benchmark | Qwen3-32B Base | Qwen2.5-32B Base | Qwen2.5-72B Base | Gemma-3-27B Base | Llama-4-Scout Base |
|---|---:|---:|---:|---:|---:|
| MMLU | **83,61** | 83,32 | 86,06 | 78,69 | 78,27 |
| MMLU-Redux | **83,41** | 81,97 | 83,91 | 76,53 | 71,09 |
| MMLU-Pro | **65,54** | 55,10 | 58,07 | 52,88 | 56,13 |
| SuperGPQA | **39,78** | 33,55 | 36,20 | 29,87 | 26,51 |
| BBH | **87,38** | 84,48 | 86,30 | 79,95 | 82,40 |
| GPQA | **49,49** | 47,97 | 45,88 | 26,26 | 40,40 |
| GSM8K | **93,40** | 92,87 | 91,50 | 81,20 | 85,37 |
| MATH | 61,62 | 57,70 | **62,12** | 51,78 | 51,66 |
| EvalPlus | **72,05** | 66,25 | 65,93 | 55,78 | 59,90 |
| MultiPL-E | **67,06** | 58,30 | 58,70 | 45,03 | 47,38 |
| MBPP | **78,20** | 73,60 | 76,00 | 68,40 | 68,60 |
| CRUX-O | **72,50** | 67,80 | 66,20 | 60,00 | 61,90 |
| MGSM | **83,06** | 78,12 | 82,40 | 73,74 | 79,93 |
| MMMLU | 83,83 | 82,40 | **84,40** | 77,62 | 74,83 |
| INCLUDE | 67,87 | 64,35 | **69,05** | 68,94 | 68,09 |

La fonte Qwen3 conta **10 vittorie su 15** per Qwen3-32B Base contro Qwen2.5-72B Base [S][1]. Questo è il dato quantitativo più importante per la priorità 32B: con una nuova generazione, passare a 72B non garantisce automaticamente qualità maggiore.

**Differenze Qwen3-32B Base rispetto a Qwen2.5-32B Base [Δ]:** MMLU-Pro `+10,44`, GPQA `+1,52`, MATH `+3,92`, EvalPlus `+5,80`, MultiPL-E `+8,76`, MMLU `+0,29`. Il vantaggio non è un effetto della quantizzazione: la tabella confronta i base model in condizioni dichiarate comuni.

### 2.3 Vincitori per sotto-compito

| Obiettivo | Primo | Evidenza | Decisione pratica |
|---|---|---|---|
| General-purpose 32B con reasoning + risposta rapida | **Qwen3-32B** | MMLU-Redux 90,9; GPQA-Diamond 68,4; AIME24 81,4 in thinking [2] | Scelta predefinita |
| Reasoning matematico/contest | **DeepSeek-R1-Distill-Qwen-32B** | MATH-500 94,3; AIME24 72,6; GPQA-D 62,1 [4] | Quando la qualità di soluzione vale la latenza |
| Coding specialist | **Qwen2.5-Coder-32B-Instruct** | HumanEval 92,7; MBPP 90,2 [6] | IDE/copilot, code repair, generazione |
| Assistente non-thinking semplice | **Qwen2.5-32B-Instruct** o Qwen3-32B `non-thinking` | Qwen2.5: MMLU-Pro 69,0, GSM8K 95,9, IFEval 79,5 [5]; Qwen3: profilo non-thinking ufficiale [2] | Qwen2.5 se si vuole comportamento più convenzionale; Qwen3 se si vuole un'unica base con entrambe le modalità. |
| Multilingue | **Qwen3-32B** | supporto dichiarato a 119 lingue/dialetti [1][2] | Preferibile per italiano + lingue diverse |

### 2.4 Punteggi vendor vs base/open-source

- La Tabella 4 [1] è **base model**, non il punteggio del chat model Qwen3-32B thinking.
- I numeri `90,9/68,4/81,4` di Qwen3-32B sono **vendor/model-card evaluation** del checkpoint post-trained, con modalità thinking e sampling specifici [2]. Sono ufficiali, ma non una verifica indipendente.
- I numeri DeepSeek-R1-Distill sono **vendor/paper evaluation** su checkpoint distillato, non un base model puro [4].
- I numeri Qwen2.5-Instruct sono **vendor evaluation** pubblicati da Qwen [5].
- I numeri Qwen2.5-Coder sono **vendor/paper evaluation**; la fonte non deve essere interpretata come leaderboard indipendente [6].

---

## 3. Ranking qualità — classe 70/72B (riferimento)

### 3.1 Ranking per profilo, non finta classifica unica

1. **Qwen2.5-72B-Instruct — miglior riferimento general-purpose documentato**. Profilo: MMLU-Pro `71,1`, MMLU-Redux `86,8`, GPQA `49,0`, MATH `83,1`, GSM8K `95,8`, HumanEval `86,6`, MBPP `88,2`, MultiPL-E `75,1`, IFEval `84,1`, Arena-Hard `81,2`, MT-Bench `9,35` [C][5]. È il confronto 70B più completo e direttamente leggibile.
2. **DeepSeek-R1-Distill-Llama-70B — primo fra i 70B per reasoning pubblicato**. AIME24 `70,0`, MATH-500 `94,5`, GPQA-Diamond `65,2`, LiveCodeBench `57,5`, Codeforces `1633` [S][4]. È un reasoning distillato, non un rimpiazzo universale di Qwen2.5-72B-Instruct.
3. **Llama 3.3 70B-Instruct — forte in instruction following/coding**. Il model card/valutazione Meta riporta MMLU `86,0`, IFEval `92,1`, HumanEval `88,4`, MBPP EvalPlus `87,6` [C][7][10]. Il confronto è meno omogeneo con la tabella Qwen2.5 perché il protocollo completo e gli split non sono identici; perciò non lo dichiaro vincitore assoluto.

Mistral Large 2 Instruct-2407 (123B) è fuori dalla classe: Mistral dichiara MMLU `84,0`, HumanEval circa `92`, GSM8K `93` [C][9]. Il suo costo memoria è quello di un modello >100B, non quello di un 70B, quindi non è una scelta hardware equivalente.

### 3.2 Quanto si guadagna passando da 32B a 70/72B?

Il confronto più onesto è **stessa famiglia, stesso tipo base model** Qwen2.5 [S][1]: Qwen2.5-72B Base rispetto a Qwen2.5-32B Base dà:

- MMLU `+2,74` punti;
- MMLU-Pro `+2,97`;
- MMLU-Redux `+1,94`;
- BBH `+1,82`;
- MATH `+4,42`;
- MGSM `+4,28`;
- INCLUDE `+4,70`;
- ma GPQA `-2,09`, GSM8K `-1,37`, EvalPlus `-0,32`, MBPP `-2,40`, CRUX-O `-1,60` e MultiPL-E `+0,40`.

Quindi il 72B offre un vantaggio medio **selettivo**, soprattutto su conoscenza/matematica/multilingue, non un raddoppio qualitativo. Il Qwen3-32B Base supera Qwen2.5-72B Base in 10/15 benchmark [1]. Per l'utente, 70B è giustificato quando il benchmark target è fra quelli in cui il 70B vince, quando il contesto/robustezza reale è più importante del throughput, o quando una singola risposta ad alto valore economico deve massimizzare la qualità.

---

## 4. Quando conviene 32B vs 70B

### Scegliere 32B

Conviene il 32B quando:

- si vogliono **inferenza locale quotidiana e LoRA/QLoRA sulla stessa piattaforma**;
- il carico è misto: chat, codice, matematica e italiano;
- si accetta di usare `Q4_K_M`, IQ4 o AWQ per inferenza e BF16/4-bit solo durante il training;
- la latenza e il costo ricorrente contano più degli ultimi 2–5 punti su un sotto-benchmark;
- si vuole conservare budget VRAM per KV cache e contesto.

Qwen3-32B è il default; Qwen2.5-Coder-32B è il default per codice; DeepSeek-R1-Distill-Qwen-32B è il default per reasoning.

### Scegliere 70/72B

Conviene il 70/72B quando:

- il modello è principalmente un **servizio di qualità**, non un laboratorio locale;
- si dispone di almeno 48–80GB effettivi per Q4 e preferibilmente 80–96GB per LoRA;
- il numero di richieste è basso ma il costo di errore è elevato;
- il benchmark target è conoscenza/multilingue/risposta generale dove il 72B mostra un vantaggio concreto;
- si usa cloud per training o inferenza, quindi il costo hardware non viene ammortizzato con molte ore locali.

Regola pratica: se il 32B supera il 70B sul benchmark del tuo caso d'uso o la differenza è inferiore a circa 3 punti e la latenza raddoppia, resta su 32B. Se la differenza è 5–10 punti sul test che decide il progetto, paga 70B solo per quella fase o usalo come teacher/evaluator.

---

## 5. Hardware per inferenza — analisi approfondita con catalogo già vagliato

> **Cosa cambia rispetto alla versione breve.** Questa sezione integra integralmente i quattro report già presenti sulla Scrivania, invece di rifare il censimento da zero. Dati, prezzi e benchmark citati come **[S]/[C]/[M]/[P]/[E]/[€]** mantengono la stessa legenda del resto del documento; dove il dato proviene da quei report è indicato esplicitamente come `cfr.` — la fonte primaria resta quella citata lì (datasheet NVIDIA/AMD/Intel/Apple, pagine Qwen, repository Geerling, listini Beam/Lambda/RunPod). Snapshot prezzi: settembre 2026, salvo diversa indicazione. Soglia “chat fluida” adottata in tutti i report: **decode ≥8 tok/s a concorrenza 1 e TTFT caldo ≤2 s su prompt 512 token** (≤5 s a 4K); 3–4 tok/s è sperimentale, non soluzione primaria [R1][R2][R3][R4].

Fonti Scrivania richiamate:
- **[R1] `REPORT-VERIFICA-INDIPENDENTE.md`** (760 righe, 09/09/2026) — verifica dual Tesla P40 2×24 GB, Q4_K_M/Q4_0/IQ4_XS verificati (47,42/41,38/39,71 GB), alternative A40/RTX A6000/3090/4090/RX 7900 XTX/P100, Mac Studio, consumi 550–750 W, 3–6 tok/s dual-P40 con ancore 3–4 e 3,3 tok/s [M].
- **[R2] `ANALISI-RIGOROSA.md`** (685 righe, 10/09/2026) — benchmark comparabili Qwen2.5-72B su A100 (11,07 tok/s GPTQ-Int4 1×A100, 11,50 AWQ, 16,47 vLLM, 46,30 vLLM 2×A100 [M]), Llama 3.1 70B su 4×H100 NIM (ITL 14,24 ms ≈70 tok/s per stream, TTFT 59,89 ms [M]), Apple Geerling (M3 Ultra 14,08 tok/s 243 W, M1 Ultra 9,84, M1 Max 7,25, Framework/Strix Halo 395 4,97 tok/s 133 W, MS-R1 0,77 [M]), TCO locale/cloud e matrice pesata.
- **[R3] `ANALISI-MINIPC-COMPONENTI.md`** (672 righe, 10/09/2026) — 16 box Mini PC/workstation compatte (MS-01 83,2 GB/s €709, UM780 XTX 89,6 GB/s €349 refurb, UM890 Pro 89,6 GB/s €489, MS-A1/A2, MS-02 102,4 GB/s, AI X1 Pro-370 89,6 GB/s, MS-S1 MAX/EVO-X2 256 GB/s €3.300–4.000, SER8, K8 Plus, GEM12…), banda calcolata `canali×MT/s×8`, previsione 70B 2,5–8 tok/s [E], TCO 3 anni a 16 h/giorno.
- **[R4] `VARIANTI-70B.md`** (352 righe, 10/09/2026) — censimento aperto 9 varianti (V1 Strix Halo 395/128 256 GB/s 4,5–8 [M/E], V2 390, V3 385, V4 HX370/890M 3,5–10, V5 7840HS/8845HS 2,5–6, V6 9955HX/285HX 4–7, V7 host+OCuLink, V8 PCIe x8/x16, V9 Apple/DGX Spark), listini CN (¥12.999–21.999, conversione ECB €1=¥7,8159 / $1,1652, stima import +15% logistica +22% IVA).

### 5.1 Budget memoria: numeri e formula (32B e 70B, dense)

Per un dense model:

- BF16: circa `parametri × 2 byte`, più runtime e KV cache;
- INT8: circa `parametri × 1 byte`, più scale/overhead;
- Q4: circa `parametri × 0,5 byte`, ma il file reale usa tipicamente 0,55–0,70 byte/parametro per scale e metadata;
- la KV cache cresce con contesto e batch.

Su **Qwen3-32B** (64 layer, 8 KV heads, head dim 128, BF16) la stima è circa **256 KiB per token** [Δ][E], prima di overhead. 32K token possono aggiungere circa 8 GiB; 128K circa 32 GiB. Su **Qwen3.8-27B** (64 layer di cui 48×Gated DeltaNet + 16×GQA 4KV, head 256) la stima vendor è **65.536 byte/token** → 16 GiB a 262K, ~61 GiB a 1M, con vantaggio strutturale ma non annullamento [22][E]. Su **Qwen2.5-72B/Llama 70B** (80 layer, GQA 8KV, hidden 8192) la KV per token è analoga per ordine di grandezza ma il file pesa ~40–47 GB (Qwen IQ4_XS 39,71, Q4_0 41,38, Q4_K_M 47,42 [R1][M]; Llama Q4_K_M 42,52, IQ4_XS 37,90 [R1][M]): il problema di fit è contemporaneamente **pesi + KV + runtime + sistema**.

Benchmark ufficiale Qwen3 su H20 misura Qwen3-32B Transformers a contesto 1 con **BF16 62.751 MB e 26,24 tok/s, FP8 33.379 MB e 7,37 tok/s, AWQ-INT4 19.109 MB e 41,8 tok/s** [M][11]. La stessa pagina riporta AWQ-INT4 SGLang `47,67 tok/s` a input 1 e `366,84 tok/s` a input 30.720, ma la definizione include prompt+generazione; non va chiamata “decode puro” [M][11]. Per 72B, Qwen pubblica su **A100 80 GB**: BF16 136,20 GB 8,73 tok/s su 2×A100, GPTQ-Int4 39,91 GB 11,07 tok/s su 1×A100, AWQ 39,44 GB 11,50 tok/s su 1×A100, vLLM GPTQ-Int4 16,47 tok/s su 1×A100 e 46,30 tok/s su 2×A100 a input 1 token [M][R2].

### 5.2 Catalogo hardware già vagliato — sinottico

La tabella seguente condensa le famiglie già misurate/preventivate nei quattro report. **Banda, TDP e prezzi sono quelli dichiarati lì**; “32B atteso” è una trasposizione [E] a parità di backend (llama.cpp/Vulkan/MLX/CUDA) perché il 32B ha ~0,44× parametri del 70B e quindi, a pari banda, attende ~1,6–2,0× tok/s. Dove esiste una misura diretta su 32B/27B è marcata [M]; altrimenti è [E] prudente e va verificata con il protocollo §5.5.

| Famiglia / architettura (report) | Box/GPU rappresentativo | RAM/VRAM | Banda teorica | Potenza/termica | Prezzo snapshot EU [€/$] | 70B Q4 decode noto [R] | 32B Q4 decode atteso — inferenza [E] | Soglia chat 8 tok/s |
|---|---|---|---|---|---:|---|---|---|
| **Dual Tesla P40** [R1] | 2×P40, PCIe 3.0 x16, CC 6.1, passiva, no NVLink, R580 ancora driver ma CUDA13 senza sm_61 prebuilt | 48 GB GDDR5 nominali (2×24) | 692 GB/s aggregati (2×346), non unificati | 500 W TDP GPU, 550–750 W wall [E][M] | ~300–450 € coppia buona; oltre 250–300 €/cad scartare | 3–6 tok/s [E]; ancore 3–4 e 3,3 su 70B [M][R1] | **6–12 tok/s** | 70B **no**, 32B **sì borderline** con IQ4_XS/Q4_0, contesto corto, split layer 1,1 |
| **Tesla P100 16GB** [R1] | 2×P100 | 32 GB HBM2 nominali | 732 GB/s per scheda 16 GB | 250–300 W/cad passiva | 80–200 $/cad | insufficiente per 70B Q4 completo | 5–9 tok/s | **no** per target |
| **RTX 3090 24GB** [R1][R2] | Singola 3090 (936 GB/s, CC 8.6) | 24 GB GDDR6X | 936 GB/s | 350 W | 550–800 € usata | 70B non entra; 2×3090 ~12,55 su Llama 3.3 70B [P] e 15–18 su issue [P][R2] | **18–35 tok/s** singola su 32B Q4 18–26 GB | **sì**, ma 70B solo in coppia |
| **Dual RTX 3090** [R1][R2] | 2×3090, 40 lane CPU X99, P2P da testare | 48 GB nominali distribuiti | — | 700 W GPU, 0,8 kW sistema | 1.100–1.600 € solo GPU | 10–18 tok/s [E/P][R2] | 12–24 tok/s in split | **sì** ma rumorosa |
| **RTX 4090 24GB** [R1] | 4090, 1 TB/s, CC 8.9 | 24 GB GDDR6X | ~1.008 GB/s | 450 W | >900–1.200 € | come 3090, 70B non entra | 22–40 tok/s | **sì** per 32B |
| **RTX 4060 Ti 16GB** [R2][R3][R4] | 4060 Ti 16GB, 128-bit, Ada 8.9, 165 W [S] | 16 GB | ~288 GB/s | 165 W | 350–550 € + dock OCuLink 100–250 € | solo split, non prova 70B | 10–20 tok/s con offload parziale | 32B **sì se** nuovo Q4 ≤15 GB (es. Qwen3.8-27B) |
| **RTX A6000 48GB** [R1][R2] | A6000 48GB ECC | 48 GB GDDR6 ECC | 768 GB/s [S] | 300 W | 2.600–3.800 USD usato [€] | fit 70B IQ4 con margine limitato | 12–22 tok/s | **sì** per 32B comodo |
| **A40 48GB** [R1] | A40 48GB ECC, passiva, 696 GB/s [S] | 48 GB | 696 GB/s | 300 W passiva | ~1.500–2.000 € [€] | idem | 10–20 tok/s | **sì** |
| **RTX 6000 Ada 48GB** | 960 GB/s ECC, 300W [S][13] | 48 GB | 960 GB/s | 300 W | ~3.000–5.000 € nuova | 70B Q4 con contesto corto | **10–22 tok/s** (tabella breve §5.2) | **sì, migliore singola per QLoRA 32B** |
| **L40S 48GB** [R2] | L40S 48GB | 48 GB | ~864 GB/s | 300 W | cloud ~$0,76/h Beam [€/$] | economico ma non ben benchmarkato su Qwen | 10–22 tok/s | **sì** |
| **A100 40/80GB** [R2] | A100 SXM 80GB, CUDA12.1, FlashAttn | 40/80 GB HBM2e | ~2.039 GB/s (80GB) | 300–400 W | cloud Beam $1,36/h, Lambda $2,79/h [€/$] | **11,07 GPTQ / 16,47 vLLM** su Qwen72B 1×A100 [M][R2] | 25–55 tok/s su 32B [E] | **sì** |
| **H100 80GB** [R2] | H100 SXM5 80GB | 80 GB HBM3 | ~3.350 GB/s | 700 W | cloud $1,83 Beam / $3,99 Lambda [€/$] | 70 tok/s per stream su **4×H100 NIM** [M][R2]; preliminare su 1×H100 ~25–40 [E] | 35–60 tok/s | **sì** |
| **RX 7900 XTX 24GB** [R1] | 7900 XTX 24GB GDDR6 | 24 GB | 960 GB/s [S] | 355 W | 600–900 $ [€] | 70B non entra intero; I-quant non compatibili Vulkan | 15–30 tok/s con ROCm/Vulkan 32B | **sì** per 32B, ecosistema meno uniforme |
| **Apple M1/M3 Ultra/Max** [R2][R3] | M1 Max 64GB 400GB/s, M1 Ultra 128GB 800GB/s, M3 Ultra 512GB | 64–512 GB UMA | 400/800 GB/s [S] | 80–243 W peak [M][R2] | — | M3 Ultra 14,08, M1 Ultra 9,84, M1 Max 7,25 su Llama 70B [M][R2][R3] | M1 Max ~12–20 su 32B [P/E] | 70B borderline su M1 Max, **sì** su Ultra |
| **Mac Studio M2 Max 64/96GB** [R1][R2][R3] | M2 Max 400GB/s [S], 32–96GB, 273GB/s per M4 Pro [S] | 64/96 GB UMA | 400 GB/s | 80 W media ipotizzata [E][R2] | ~1.500 € usato [€] | ipotizzato 6–10 su 70B [E][R1] — non validato su target esatto | **14–28 tok/s** su 32B [E] | **sì** per 32B |
| **Mac Studio M4 Max 128GB / M5** [R1][R3] | M4 Max 128GB, M5 Max 614GB/s, M5 Ultra 1,2 TB/s, 480W max continuo [S] | 128 GB UMA | fino 1,2 TB/s | <480W | ~3.500 $ (128GB config) [€/$] | ancora da benchmarkare su target [P] | ~16–30 tok/s | **sì** |
| **Strix Halo 395 128GB** [R3][R4] | Ryzen AI Max+395, 16C/32T Zen5, 8060S 40CU, LPDDR5X-8000 256-bit | 128 GB UMA saldata | **256 GB/s** (8000×32) [S/C] | 140 W media ipotizzata (130 sust/160 peak) [E][R3] | EU €3.300–4.000 (MS-S1 MAX), €3.230 (EVO-X2), CN ¥12.999 GTR9 / ¥13.999 FAEX1 / ¥14.499 EVO-X2 / ¥14.798 H1 [€][R4] | **4,97 tok/s** Llama70B su Framework 395/128 [M][R2][R3][R4]; AMD claim 15 tok/s non omogeneo [S/P] | **8–18 tok/s** su 32B Q4 [E] (riserva banda) | 70B **borderline no**, 32B **sì** |
| **Strix Halo 390/385 128GB** [R4] | 390 12C/8050S 32CU, 385 8C/32CU stesso bus | 128 GB | 256 GB/s | simile, meno CU | spesso più economico CN | nessuna misura specifica | 6–14 tok/s | 32B **sì** |
| **DDR5 dual-channel Mini PC** [R3][R4] | Ryzen 7840HS/7940HS/8845HS/8945HS (780M 12CU) 2×SO-DIMM 5600; Intel 13900H 5200 (83,2 GB/s) | 64/96GB (2×48 da verificare su UM790/SER8) | 83,2–89,6 GB/s [C] | 30–65 W box | UM780 XTX €349 refurb/€650 config; UM890 €489; K8 €400–550; GEM12 $320 barebone [€] | 70B CPU-only 2,5–4,5; con 780M 3–6 [E][R3]; con OCuLink 3–8 [E] | **5–11 tok/s** su 32B CPU/780M | 32B **sì** per chat con Q4 leggero (27B), borderline su 32,8B Q4 22–26GB |
| **HX370 96/128GB + 890M/OcuLink** [R3][R4] | AI X1 Pro-370 12C/24T, 890M 16CU, 2×SO-DIMM 5600 =89,6GB/s; EVO-X1 32GB LPDDR5X 120GB/s **non fit 70B** | 96GB max pratico / 32GB saldato da scartare | 89,6 / 120 | 70W box [E] | AI X1 Pro €729 barebone/€1.639 64GB/€1.800 96GB; EVO-X1 €1.030 32GB [€] | CPU 3,5–5,5, 890M 4–8, +OCuLink 5–10 [E][R3] | **7–15 tok/s** su 32B | 32B **sì**, 70B no senza eGPU |
| **9955HX / 285HX workstation** [R3] | 9955HX 16C Zen5 610M 2CU 89,6GB/s; 285HX 24C/102,4GB/s DDR5-6400 4 slot 2ch | 96–256GB | 89,6 / 102,4 | 75–90 W | MS-A2 €839 base; MS-02 ~$599–$1.159 [€] | CPU 4–7 su 70B [E][R3] | 8–16 tok/s su 32B | **sì** per 32B, ma valore come host PCIe |
| **Host + OCuLink / PCIe + GPU** [R2][R3][R4] | OCuLink PCIe4.0×4 ~8GB/s tipici [S/P]; PCIe4.0×8 (MS-01) /×16 (MS-02) | 96GB + 16–24GB | link limitato | +dock 100–250€ | host €570–900 + GPU | 70B 5–12 con GPU moderna [E][R4] | **12–28 tok/s** su 32B offload | 70B **forse**, 32B **sì** |
| **DGX Spark GB10** [R4] | GB10 128GB LPDDR5X 256-bit 273GB/s [S] | 128 GB | 273 GB/s | 240W PSU | non economico | NVIDIA dichiara 200B max, non decode Qwen72 [S] | ~18–32 tok/s | **sì** |

> **Lettura chiave per il 32B.** La soglia 8 tok/s, fallita da quasi tutti i Mini CPU-only sul 70B [R2][R3], diventa raggiungibile sul 32B già con DDR5 dual-channel + 780M/890M e con qualsiasi 24/48GB discreta, perché il file Q4 passa da 39–47 GB (70B) a **22–26 GB (32,8B)** e a **15,93 GiB (+0,87) per Qwen3.8-27B** [22][M]. Per Qwen3.8-27B Q4+KV intesa: su 24GB discreti resta margine per 4–8K contesto; su 48GB per 32K comodo; su Strix/Apple 128GB per 262K nativo solo con KV ottimizzata.

### 5.2 Hardware minimo/ideale per ciascun modello 32B

Le velocità locali sotto sono **intervalli di progetto [E]** salvo dove marcato [M] (Qwen H20, Geerling, NIM). Dipendono da backend (llama.cpp/Vulkan/ROCm/MLX/CUDA/SGLang), quantizzazione (GGUF Q4_K_M/IQ4_XS/AWQ), contesto, offload, temperatura, batch e token di reasoning (CoT). Il benchmark Qwen H20 [11] resta la misura pubblica più utile, non una promessa per ogni GPU.

| Modello | Inferenza minima accettabile (32B/Q4, contesto 4–8K) | Inferenza ideale | Memoria consigliata (Q4) | Decode atteso su 32B [E/M] | Consumo indicativo |
|---|---|---|---|---:|---:|
| Qwen3-32B (32,8B, 64L, 8KV, 256 KiB/tok) | GPU 24GB con Q4/AWQ, 4–8K; oppure Mini PC 64GB UMA + 780M/Vulkan; oppure Mac 64GB; oppure host+OCuLink+16GB per split | RTX 5090 32GB Q4/AWQ per velocità; RTX 6000 Ada 48GB ECC per stabilità/ECC; Strix Halo 395 128GB per contesto/offload senza discreta; H20/A100/H100 per server | Q4 22–26GB con contesto corto; 48GB per 32K comodo; BF16 ~62,75GB solo pesi + 8GB KV@32K + runtime → ~75GB [C][M][11] | RTX 5090 18–35; RTX 6000 Ada 10–22; Strix 395 8–18 su 32B (4,97 su 70B proxy [M]); Apple M1 Max ~12–20 [P/E]; H100 35–60 [E] | RTX 5090 575W TGP [S][12]; RTX 6000 Ada 300W [S][13]; Strix mini 80–140W [E]; M1 Max 7–15W chip + sistema |
| DeepSeek-R1-Distill-Qwen-32B (reasoning) | GPU 24GB Q4 ma 32GB più sicuro a causa CoT | A100/H100 80GB o RTX 5090 32GB con KV grande | Q4 22–28GB + KV CoT; pianificare 48–80GB ideale per dialoghi lunghi | RTX 5090 15–30; RTX 6000 8–18; H100 30–55 [E] — ma token/risposta 3–8× per via CoT [P] | come Qwen3; costo per risposta ∝ token generati |
| Qwen2.5-32B-Instruct | GPU 24GB Q4, 4–8K | RTX 5090 32GB o RTX 6000 Ada 48GB; A100/H100 se server | Q4 22–26GB; 48GB per 32K; BF16 ~75GB | RTX 5090 18–35; RTX 6000 10–22; H100 35–60 [E] | 0,6–0,9kW sistema 5090 [E] |
| Qwen2.5-Coder-32B-Instruct | GPU 24GB Q4 per prompt brevi; 32GB se repo+contesto | RTX 5090 32GB workstation; RTX 6000 Ada 48GB ECC per uso continuo | Q4 22–28GB; 48GB per repository ampio | RTX 5090 18–35; RTX 6000 10–22; H100 35–60 [E] | 300W RTX6000; 575W RTX5090 [S][12][13] |
| **Qwen3.8-27B (27,78B, 65,5 KiB/tok, 262K nativo)** [22] | **GPU 24GB Q4 15,93 GiB** [M] già comoda a 4–8K; Mini PC 64–96GB DDR5 + 780M già sopra soglia su 32B | RTX 6000 Ada 48GB o 5090 32GB; Strix 128GB per 262K con KV 16GB @262K (67,76 GiB min pesi+KV [E]) | Q4 15,93+0,87 [M][22]; FP8 28,76 GiB [M]; 24GB ok moderato, 48GB per 262K | 20–38 su 5090; 10–20 Strix; 35–60 SGLang H100 [E] | come sopra, meno VRAM per pesi |
| Gemma 3 27B IT (25,6B non-emb, 128K, 5:1 local/global sw 1024) | GPU 24GB Q4 ~14–16GB | Strix 128GB o Apple 64GB per 32K+; RTX 6000 48GB ideale server | BF16 54,0GB / int4 14,1GB +KV 72,7GB@32K [M][24]; Q4 ~14–16GB terze parti | 18–35 su 5090 [E] | simile |
| GLM-4.7-Flash 30B-A3B (MoE 3B attivi, 128K) | GPU 22–26GB Q4 | 5090 32GB | ~14–16GB Q4 [E] | 25–45 su 5090 [E] (3B attivi → 1,5–2× vs dense) | 300–575W |

**Note comuni già validate nei report:**
- **RTX 5090:** 32GB GDDR7, MSRP $1.999, TGP 575W, PSU 1.000W [S][12]. Singola 5090 eccellente per Q4 e QLoRA stretto 32B, non per BF16 completo né per inferenza+training simultanei.
- **RTX 6000 Ada:** 48GB GDDR6 ECC, 960GB/s, 300W [S][13]. Più lenta della 5090 in tok/s ma migliore per QLoRA 32B comodo (48GB, ECC, driver workstation).
- **Strix Halo 395 128GB:** AMD dichiara max 128GB LPDDR5X-8000 e bus 256-bit [S][14]. È UMA, non VRAM discreta; consente Q4/BF16 parziale e coesistenza col sistema, ma banda/driver ≠ NVIDIA. Attesa prudente **8–18 tok/s su 32B Q4** [E] vs 4,97 misurati su 70B proxy [M][R2–R4]. Prezzi: GMKtec 128GB/2TB EU ~€1.959,99 [€][15]; GTR9/FAEX1/H1 CN ¥12.999–14.798 → €2,3–2,6k import stimato (+15% +22% IVA) [R4][C].
- **Apple Silicon:** Mac Studio M4 Max fino a 128GB UMA [S][16]; MLX/llama.cpp per inferenza, training via MLX/MPS sperimentale vs CUDA. Silenziosa/capiente, non per tok/s training.
- **Dual-P40 / dual-3090:** vedi riga catalogo e [R1][R2]; P40 richiede toolchain congelato `sm_61`, llama.cpp con `--split-mode layer --tensor-split 1,1 --n-gpu-layers all --ctx-size 4096 --parallel 1`, verifica P2P `GGML_CUDA_P2P=1` e no vLLM/SGLang moderni (CC≥7.5 / sm75+) [R1].

### 5.3 70/72B in inferenza — dettaglio con benchmark ufficiali comparabili

Stessa legenda; qui la soglia è più severa e i numeri sono quelli che hanno generato le decisioni nei report [R2].

| Modello | Q4 minimo (GGUF verificato) | Ideale | Memoria pratica (pesi+KV+runtime) | Decode atteso — evidenza comparabile | Consumo |
|---|---|---|---|---:|---:|
| Qwen2.5-72B-Instruct | 48GB con IQ4_XS 39,71 / Q4_0 41,38 e contesto corto; Q4_K_M 47,42 troppo vicino a 48 nominali [M][R1] | 2×48GB, A100/H100 80GB, oppure 128GB UMA | Q4 40–48GB + KV; 80GB solo moderato; BF16 136,20GB su 2×A100 [M][R2]; divisione tra GPU + buffer per device | 1×A100 GPTQ-Int4 11,07, AWQ 11,50, vLLM 16,47; 2×A100 vLLM 46,30 [M][R2]; 2×3090 10–18 [P][R2]; 2×P40 3–6 [E][R1] | 0,8–1,3kW GPU+host per 2×48GB [E] |
| Llama 3.3/3.1 70B-Instruct | Q4_K_M 42,52 / IQ4_XS 37,90 [M][R1] | 80GB GPU o 2×48GB, oppure Strix 128GB | Q4 42GB + KV; BF16 ~140GB | 4×H100 NIM 70 tok/s per stream, TTFT 59,89 ms (1000/1000) e 3.794 tok/s aggregati a conc.100 [M][R2]; M1 Max 7,25, M1 Ultra 9,84, M3 Ultra 14,08 [M][R2]; Strix 395 4,97 [M][R3] | 0,5–1,0kW [E] |
| DeepSeek-R1-Distill-Llama-70B | 48GB Q4 solo corto; meglio 80GB | A100/H100 80GB o 2×48GB | CoT lungo → 64–96GB | 5–14 tok/s ma token/risposta 3–8× [E][P] | come sopra |

Per un 70B la memoria Q4 non è il solo problema: un ragionamento di 8–16K token consuma parecchi GB di KV e riduce la concorrenza. Un 32B Q4 su 32GB è molto più pratico di un 70B Q4 che “entra appena” in 48GB — è precisamente il motivo del verdetto [R1]: **dual-P40 promossa solo come low-cost condizionata (300–450 € coppia testata, reso, airflow, 3–4 tok/s accettati) e respinta come scelta predefinita 16 h/giorno** [R1].

### 5.4 Quando una famiglia conviene — lettura per il 32B (vs 70B)

- **Budget minimo e silenzio, 32B Q4:** Mini PC DDR5 dual-channel 89,6 GB/s con 64/96GB (UM780 XTX refurb €349/€650 config, UM890 €489, K8 €400, GEM12 $320 barebone) + 780M [R3][R4] — **insufficiente per 70B chat** [R2][R3] ma **sufficiente/borderline per 32B** (5–11 tok/s [E]) e ottimo host per modelli 7–20B, embedding e draft. Se già posseduto, aggiungere QLoRA cloud.
- **Host economico per eGPU:** gli stessi box con **OCuLink** (UM780 XTX, UM890 Pro, K8 Plus, GEM12 Pro, AI X1 Pro-370) [R3][R4] — non validati a 8 tok/s su 70B [R2], ma su 32B la GPU da 16GB (4060 Ti 165W) basta per tenere il file 22–26GB e superare la soglia con split layer; testare PCIe 4.0×4 negoziato, non fallback USB4/Gen3.
- **Workstation compatta con PCIe:** MS-01 x8, MS-A2 x8 split, MS-02 x16 PCIe5.0 fino a 256GB ECC [R3][R4] — CPU-only 4–7 tok/s su 70B [E], quindi non promosse CPU-only; con GPU discreta diventano workstation vere (8+ tok/s 70B possibile [E] ma 700–800W e rumore).
- **Strix Halo 395 128GB 256GB/s:** unica UMA x86 compatta con banda e iGPU coerenti [R3][R4]; 70B 4,97 tok/s non passa la soglia [M][R2][R3] e prezzo €3.300–4.000 è alto per il solo 70B; **su 32B è invece promossa** (8–18 tok/s [E]) e giustificata se servono anche 100–128B locali, privacy e silenzio.
- **Apple Silicon 64/96/128GB UMA 400–800 GB/s:** migliore locale silenziosa già pronta; 70B su M1 Max 7,25 sotto soglia, su M1 Ultra/M3 Ultra sopra [M][R2]; su 32B tutti sopra soglia con margine. Mac mini M4 Pro 48GB non robusto per 70B Q4_K_M [R2] ma ok per 32B.
- **GPU singola 24GB (3090/4090/7900 XTX):** eccellenti per 32B (18–40 tok/s), insufficienti da sole per 70B Q4 senza offload/coppia [R1].
- **GPU singola 48GB ECC (A6000/A40/6000 Ada/L40S):** vera alternativa architetturale [R1] — eliminano split, Ampere/Ada CC 8.6/8.9, supporto moderno; per 32B sono il punto ideale capacità/affidabilità/QLoRA; per 70B solo quant con margine (Qwen Q4_K_M troppo stretto su 48 [R1]).
- **Dual-P40 / Dual-3090:** “miglior VRAM/€” non equivale a “migliore decisione complessiva” [R1][R2]; P40 lenta/legacy (vLLM/SGLang non supportati CC6.1), 3090 veloce ma 700W/rumore [R1][R2]. Per 32B una singola 24GB è già preferibile alla coppia legacy.
- **Cloud A100/H100/L40S/A6000 sempre caldo vs intermittente:** Qwen 11–16 tok/s su 1×A100 72B [M][R2] passa la soglia chat a contesto corto, ma costo 16h/giorno 7–22k€/anno (Beam) vs TCO locale [R2][R3]; sotto ~100–150 h/mese l’ibrido (GPU locale 16–24GB per 7–32B + cloud on-demand per 70B) domina economicamente [R2][R3].

### 5.5 Protocollo di accettazione prima di dichiarare “gira”

Nessuna tabella sostituisce una misura sul file e sul backend dell’utente [R1][R2][R3]. Per ogni candidato valga lo stesso test:

1. File preciso annotato (es. `Qwen2.5-32B-Instruct-Q4_K_M.gguf` o `Qwen3.8-27B-Q4_K_M` 15,93 GiB [22][M], hash, quant con/senza imatrix).
2. Runtime e commit (llama.cpp/Vulkan/ROCm/Metal/CUDA, versione, `metal`/`vulkan`/`cuda` backend, `llama-bench`/`llama-server` flags).
3. Contesto 4.096 (poi 8.192), un solo slot, batch documentato, 1.000 token dopo warm-up, tre run (media/min/max).
4. Misurare separatamente **prefill tok/s, TTFT P50/P95 caldo** (512 e 4K prompt), **decode tok/s e ITL P95**, VRAM/RAM libera, watt alla presa (idle/prefill/decode), temperatura dopo 30–60 min e rumore a 50 cm [R2][R3].
5. Test misto: job async 2.000 token + chat 512 token dopo 10 s (ripetere 5×); promuovere solo se TTFT P95 ≤5 s e decode chat ≥8 tok/s [R2][R3].
6. Per P40/dual-GPU: `--split-mode layer --tensor-split 1,1 --n-gpu-layers all --ctx-size 4096 --parallel 1` come baseline [R1]; P2P solo dopo baseline; KV q8 solo con layer mode.
7. Per Mini PC DDR5: verificare 2×48GB JEDEC, 5600/6400 mantenuti, dual-channel, Memtest, banda STREAM [R3].
8. Per OCuLink/eGPU: verificare negoziazione PCIe 4.0×4, PSU adeguato, decode a 0/metà/tutti i layer offloadati [R3][R4].


---

## 6. Fine-tuning LoRA/QLoRA sui 32B — tutti gli scenari

### 6.1 Scelte tecniche comuni

**LoRA** congela i pesi e allena matrici a basso rango; il lavoro originale mostra una grande riduzione dei parametri allenabili e della memoria rispetto al full fine-tuning [S][17]. **QLoRA** conserva il modello base in 4-bit, propaga il gradiente attraverso il base congelato e allena l'adapter; il paper dimostra 65B su una singola GPU da 48GB con NF4, double quantization e paged optimizer [S][18].

Configurazione iniziale prudente per un 32B:

```text
LoRA:  bf16 base, r=16, alpha=32, dropout=0.05,
      target q_proj,k_proj,v_proj,o_proj e gate/up/down_proj se la VRAM lo consente.
QLoRA: NF4 + double quant, compute bf16, r=16 (r=32 solo dopo validazione),
       paged AdamW 8-bit, gradient checkpointing, packing controllato.
SFT:   micro-batch 1; gradient accumulation per batch effettivo 8–32;
       learning rate LoRA 1e-4–2e-4; 1–3 epoche, early stopping.
```

PEFT/Transformers conferma che solo i parametri dell'adapter sono aggiornati, che i checkpoint contengono l'adapter e non il base model, e che QLoRA può essere caricato con base quantizzato e `device_map="auto"` [S][19]. Per Qwen3, mantenere il chat template e separare esempi `/think` e `/no_think`: mescolare risposte con CoT e risposte rapide senza etichetta può degradare il controllo della modalità [S][2].

Stime per **SFT supervised**, non pre-training né RL:

| Metodo 32B | Memoria minima teorica/pratica | Setup sostenibile | Batch/seq len realistici | Training throughput [E] |
|---|---:|---:|---|---:|
| LoRA BF16 | circa 80GB con checkpointing e seq 512–1K; 96GB raccomandati | A100 80GB solo configurazione stretta; H100 80/96GB ideale; 2×48GB con FSDP | micro-batch 1, seq 512–2K; grad accumulation 8–32 | A100 1–3 tok/s; H100 2–6 tok/s [E] |
| QLoRA NF4 | 28–32GB per seq 512–1K molto ottimizzato; 40–48GB raccomandati | RTX 5090 32GB solo dataset/seq prudenti; RTX6000/L40S 48GB ideale; A100 40/80GB molto comoda | micro-batch 1, seq 512–2K su 32GB; seq 2–4K su 48/80GB | RTX5090 1–4; 48GB 1–4; A100 2–6; H100 4–10 [E] |
| QLoRA CoT/reasoning lungo | 40–48GB minimo realistico per 2K; 80GB per 4–8K | A100/H100 80GB | micro-batch 1, seq 2–8K, grad accumulation 16–64 | fortemente dipendente dalla lunghezza [E] |

Non esiste una VRAM “minima” indipendente da sequence length: attivazioni, KV/attention e optimizer possono far esplodere il picco. Un QLoRA che parte su 24GB a seq 256 non è prova che un SFT a seq 4096 funzionerà.

### Scenario 1 — stessa macchina: inferenza + training insieme

#### Opzione A: Strix Halo 395, 128GB unified memory

- **QLoRA:** praticabile come laboratorio locale; riservare circa 40–60GB al training 32B e 25–35GB all'inferenza Q4, lasciando margine al sistema [E].
- **LoRA BF16:** possibile solo con seq corto e offload aggressivo; non la scelta affidabile per lavoro continuativo.
- **Coesistenza reale:** inferenza Q4 e QLoRA contemporanei possono stare nei 128GB, ma condividono banda memoria e CPU/GPU; aspettarsi forte calo del tok/s e maggiore latenza [E].
- **Batch/seq:** training micro-batch 1, seq 512–1K; inferenza 4–8K. Per 32K o CoT lungo, eseguire a turni.
- **Training token/s:** circa 0,2–1,0 token/s per il job QLoRA, stima conservativa [E]; non confonderlo con decode inference.
- **Consumo/costo:** mini-PC 80–140W tipici [E], prezzo di riferimento circa €1.960 per GMKtec 128GB/2TB (listino settembre 2026) [€][15].
- **Pro:** una sola macchina, 128GB, silenziosa, nessun cloud, può tenere base, adapter e servizio.
- **Contro:** driver/stack meno maturo, banda condivisa, throttling termico, training molto più lento di CUDA, impossibile mantenere latenza stabile mentre si allena.
- **Quando sceglierlo:** prototipi, piccoli dataset, adapter occasionali, privacy e disponibilità 24/7; non per molte iterazioni su dataset grandi.

#### Opzione B: RTX 5090 32GB

- **QLoRA:** possibile solo con NF4, paged optimizer, checkpointing, seq 512–1K, micro-batch 1; 32GB è una soglia stretta [E].
- **LoRA BF16:** non pratico su singola scheda: i soli pesi BF16 del 32B richiedono circa 65GB, prima delle attivazioni [E].
- **Inferenza + training simultanei:** sconsigliati; il modello Q4 può occupare 20–26GB e lasciare troppo poco al training.
- **Throughput training:** circa 1–4 tok/s [E], spesso più veloce del mini-PC ma senza headroom.
- **Consumo/capex:** GPU 575W e PSU 1.000W [S][12]; MSRP GPU $1.999 [€/$][12], workstation completa realisticamente da quotare separatamente.
- **Pro:** velocità inference molto superiore a unified memory, ecosistema CUDA, ottima per Q4 e piccoli QLoRA.
- **Contro:** 32GB non bastano per LoRA BF16, rumore/calore, nessuna separazione fra servizio e job, rischio OOM.
- **Quando sceglierlo:** se il fine-tuning è occasionale e si accetta di fermare il servizio durante il training.

#### Opzione C: Mac Studio M4 Max 128GB

- **QLoRA/LoRA:** inferenza Q4 molto comoda per memoria; training via MLX/PyTorch-MPS da considerare sperimentale rispetto a CUDA. Verificare supporto esatto della versione di `mlx-lm`/Transformers prima dell'acquisto.
- **Coesistenza:** capienza sì, latenza stabile no; condividere 128GB fra processo di training, base e servizio causa pressione di memoria e swap.
- **Throughput:** 3–10 tok/s Q4 32B e 0,2–1 tok/s training sono intervalli prudenti [E], non dati Apple ufficiali.
- **Costo:** Apple supporta 128GB unified; configurazioni 128GB sono state quotate intorno a $3.499–$3.699 (listini 2025) [€/$][16].
- **Pro:** efficienza, silenzio, unified memory, ottima macchina da inferenza locale.
- **Contro:** training meno standardizzato, nessuna CUDA, costo elevato per tok/s, rischio incompatibilità kernel.
- **Quando sceglierlo:** priorità inferenza/desktop e training raro, non se LoRA è requisito principale.

#### Verdetto Scenario 1

Per la richiesta “inferenza e allenamento insieme”, la sola configurazione realmente equilibrata è **Strix Halo 128GB + QLoRA**, accettando throughput basso. Una 5090 è migliore se si può fermare l'inferenza durante il training; per LoRA BF16 affidabile serve cloud o almeno 80–96GB.

### Scenario 2 — cloud per training, inferenza locale

Separare i due carichi elimina la competizione di VRAM e consente di usare un modello Q4 locale mentre il cloud allena l'adapter. Dopo il training si scaricano solo i pesi LoRA, normalmente da decine a centinaia di MB o pochi GB, non il base model [S][19].

#### GPU cloud consigliate

| GPU cloud | VRAM | LoRA BF16 32B | QLoRA 32B | Giudizio |
|---|---:|---|---|---|
| A100 40GB | 40GB | stretto/non consigliato | sì, seq 512–1K | minimo cloud economico |
| A100 80GB | 80GB | sì con checkpointing e seq 512–2K; 96GB meglio | sì, seq 2–4K | **miglior rapporto costo/rischio** |
| RTX 6000 Ada | 48GB ECC | stretto; no per seq lunga | sì, comodo | ottima singola workstation cloud |
| L40S | 48GB | stretto | sì, comodo | alternativa alla RTX6000; verificare kernel/host |
| H100 80GB | 80GB | sì, più veloce; ideale per CoT | sì, molto comodo | massimo throughput, costo maggiore |

Snapshot di prezzo Lambda consultato a settembre 2026: A100 SXM 80GB **$2,79/GPU-ora**, H100 SXM 80GB **$3,99/ora**, A6000 48GB **$1,09/ora**, Quadro RTX 6000 24GB **$0,69/ora** [€/$][20]. La RTX 6000 Ada 48GB e L40S sono spesso quotate circa $0,8–1,5/ora nei listini spot/on-demand, ma il valore cambia per provider e disponibilità [E]; non uso questo intervallo come prezzo garantito.

#### Costo per sessione di fine-tuning

Assunzione trasparente: **8 ore di GPU + 2 ore equivalenti** per installazione, download, valutazione e checkpoint, cioè 10 ore fatturate; esclusi storage, VAT, egress, notebook inattivo e dataset [E].

| Configurazione | Prezzo unitario usato | Costo GPU 10h | Sessione realistica |
|---|---:|---:|---:|
| A100 80GB | $2,79/h [€/$][20] | $27,90 | $30–45 includendo storage/overhead [E] |
| H100 80GB | $3,99/h [€/$][20] | $39,90 | $45–65 [E] |
| A6000 48GB | $1,09/h [€/$][20] | $10,90 | $13–25 [E] |
| RTX6000/L40S 48GB | $0,8–1,5/h [E] | $8–15 | $12–30 [E] |

Un dataset di 50–200K esempi brevi può richiedere meno di 8 ore su H100 ma più giorni su hardware unified; il costo va calcolato come `ore_gpu × tariffa`, non come “costo del modello”. Per DeepSeek-R1-Distill e dati CoT lunghi, pianificare il doppio delle ore o ridurre la seq length [E].

#### Pro/contro e scelta

- **A100 80GB:** pro: costo, memoria, ecosistema stabile, LoRA BF16 possibile; contro: meno veloce dell'H100. È la scelta standard.
- **H100 80GB:** pro: meno tempo, migliore per seq lunga e molti esperimenti; contro: il sovrapprezzo non migliora il punteggio del modello, solo il tempo. Sceglierlo per iterazioni rapide o dataset/CoT grandi.
- **RTX6000 Ada/L40S 48GB:** pro: costo basso e sufficiente per QLoRA; contro: LoRA BF16 32B è stretto e la velocità è inferiore a A100/H100. Sceglierla per QLoRA SFT ordinario.
- **A100 40GB/24GB:** pro: costo minimo; contro: rischio OOM e contesto corto. Solo QLoRA prudente.
- **Inferenza locale:** il servizio resta su Strix/Apple/5090, mentre il cloud restituisce adapter. Pro: privacy dei prompt in produzione e costi cloud solo durante i picchi. Contro: occorre versionare base model, tokenizer, chat template e adapter.

### Scenario 3 — ibrido economico ottimale

Configurazione consigliata:

1. **Locale:** Strix Halo 395 128GB oppure GPU già posseduta, per inferenza Q4, test regressivi e dataset curation.
2. **Cloud on-demand:** A100 80GB per QLoRA/LoRA ordinario; H100 solo per CoT lungo, batch più alto o iterazioni urgenti.
3. **Artefatti:** salvare dataset hash, commit del tokenizer/chat template, `adapter_config.json`, metriche e seed; scaricare solo adapter e checkpoint migliori.
4. **Servizio:** tenere Qwen3-32B Q4 locale; attivare Qwen3 `thinking` solo per richieste difficili. Usare DeepSeek-R1-Distill come modello secondario per valutazione matematica, non necessariamente come endpoint principale.

**Pro:** CAPEX minimo, cloud pagato solo quando serve, possibilità di testare molti rank/learning rate, inferenza privata locale.  
**Contro:** tempi di upload/download, gestione segreti e versioni, dipendenza dalla disponibilità GPU.  
**Costo realistico:** se si fanno 2 sessioni A100 da 10 ore al mese, circa **$56 di GPU [€/$]** più storage/IVA [Δ][20]; è molto inferiore all'acquisto di una GPU workstation 48GB se il training è occasionale. Se il training supera circa 300–500 ore/anno, rivalutare una GPU locale o un server dedicato [E].

### 6.2 Rischi di overfitting, qualità e throttling

- **Overfitting:** su dataset piccolo, partire da 1 epoca, validazione separata, early stopping e rank 8/16. Non usare HumanEval/MMLU come unico criterio: misurare anche un set privato.
- **Catastrophic forgetting:** per Qwen3 mantenere esempi generali e campioni sia `think` sia `no_think`; per Coder mantenere una quota di testo/math se serve comportamento general-purpose.
- **Contaminazione:** non usare domande pubbliche dei benchmark come training set; AIME e HumanEval sono particolarmente sensibili.
- **Seq length:** raddoppiare la sequenza può aumentare più che linearmente il picco di memoria. Ridurre prima `micro_batch`, poi seq, poi target modules.
- **Throttling:** 5090 a 575W e mini-PC in carico continuo richiedono ventilazione; su unified memory il throttling riduce insieme inferenza e training. Per sessioni >2–4 ore monitorare temperatura, clock, power limit e tok/s [E].
- **Quantizzazione dell'inferenza:** fare benchmark dell'adapter sulla stessa quantizzazione usata in produzione. Un adapter addestrato su BF16 non garantisce identica qualità quando applicato a GGUF Q4; confrontare base Q4, adapter Q4 e merge BF16.

---

## 7. Matrice di sintesi

| Modello | Punteggio qualità documentato | VRAM inferenza | Tok/s decode atteso | VRAM LoRA | Costo minimo | Giudizio |
|---|---|---:|---:|---:|---|---|
| **Qwen3-32B thinking** | LiveBench 74,9; GPQA-D 68,4; MMLU-Redux 90,9; AIME24 81,4 [C][2] | Q4 22–26GB; 48GB comoda; BF16 ~75GB | 18–35 su 5090; 8–18 Strix [E] | LoRA 80–96GB; QLoRA 32–48GB | Strix QLoRA locale; A100 80 cloud | **Raccomandato come default** |
| **DeepSeek-R1-Distill-Qwen-32B** | AIME24 72,6; MATH-500 94,3; GPQA-D 62,1; LiveCode 57,2 [S][4] | Q4 22–28GB + KV CoT; 48–80GB ideale | 15–30 su 5090; 30–55 H100 [E] | LoRA 80–96GB; QLoRA 40–80GB per CoT | QLoRA A100 80 | reasoning eccellente, latenza elevata |
| **Qwen2.5-32B-Instruct** | MMLU-Pro 69,0; GSM8K 95,9; MATH 83,1; HE 88,4; IFEval 79,5 [C][5] | Q4 22–26GB; 48GB per contesto | 18–35 su 5090 [E] | LoRA 80–96GB; QLoRA 32–48GB | 5090 QLoRA stretto o A100 | generalista maturo e facile da adattare |
| **Qwen2.5-Coder-32B-Instruct** | HumanEval 92,7; MBPP 90,2; Aider 73,7 [C][6] | Q4 22–28GB; 48GB per repository | 18–35 su 5090 [E] | LoRA 80–96GB; QLoRA 32–48GB | RTX6000/L40S QLoRA | **migliore per coding specialist** |
| **Qwen2.5-72B-Instruct** | MMLU-Pro 71,1; MATH 83,1; GSM8K 95,8; IFEval 84,1; Arena-Hard 81,2 [C][5] | Q4 48–64GB; BF16 145GB | 7–16 su 2×48GB; 12–25 H100 [E] | LoRA 160–192GB; QLoRA 80–96GB | A100/H100 80 per QLoRA | riferimento qualità, non acquisto locale primario |
| **Llama 3.3 70B-Instruct** | MMLU 86,0; IFEval 92,1; HE 88,4 [C][7][10] | Q4 48–64GB; 80GB GPU preferibile | 7–16 2×48GB; 12–25 H100 [E] | LoRA 160–192GB; QLoRA 80–96GB | A100/H100 80 | forte su instruction following |
| **DeepSeek-R1-Distill-Llama-70B** | AIME24 70,0; MATH-500 94,5; GPQA-D 65,2; LiveCode 57,5 [S][4] | Q4 48–64GB + KV CoT | 5–14 [E] | QLoRA 80–96GB; LoRA multi-GPU | A100/H100 80 | teacher/reasoning di riferimento |

**Interpretazione del costo minimo:** “locale” significa che il modello entra in memoria con Q4 ma non implica LoRA BF16; “cloud” indica una GPU con headroom per SFT. I costi sono per la piattaforma, non per il download dei pesi.

---

## 8. Raccomandazione finale per l'utente

### Modello da scegliere

**Scegliere Qwen3-32B come modello primario.** La motivazione quantitativa è:

- nel confronto base omogeneo, Qwen3-32B supera Qwen2.5-32B su MMLU-Pro di `+10,44` punti, EvalPlus di `+5,80` e MultiPL-E di `+8,76` [Δ][1];
- supera Qwen2.5-72B Base in 10/15 benchmark, quindi il passaggio a 70B non è automaticamente un miglioramento [S][1];
- nel checkpoint post-trained thinking, il profilo documentato è GPQA-D `68,4`, MMLU-Redux `90,9`, AIME24 `81,4` e LiveBench `74,9` [C][2];
- può spegnere il reasoning senza cambiare famiglia/checkpoint, riducendo la latenza [S][2];
- il modello ha quantizzazione AWQ ufficiale con footprint H20 misurato di circa 19,1GB a contesto 1 [M][11], quindi è molto più semplice da distribuire di un 70B.

**Eccezioni:**

- se il 70–80% del lavoro è codice, scegliere **Qwen2.5-Coder-32B-Instruct**;
- se il lavoro è matematica/reasoning e la risposta può attendere, provare **DeepSeek-R1-Distill-Qwen-32B**;
- se il benchmark interno dimostra un vantaggio stabile del 70B superiore a circa 5 punti, usare Qwen2.5-72B o R1-Distill-Llama-70B in cloud, non costruire subito una workstation 70B.

### Hardware consigliato

1. **Se la priorità è usare la macchina già discussa, il miglior acquisto capace di fare entrambe le cose è Strix Halo 395 con 128GB unified memory.** Usarlo per Qwen3-32B Q4, curation e QLoRA con seq 512–1K; fare training e inferenza contemporaneamente solo per prototipi. Il vantaggio è la capienza; lo svantaggio è la velocità e la banda condivisa.
2. **Se la priorità è velocità inference**, scegliere una **RTX 5090 32GB** con Q4/AWQ. È la scelta ideale per decode locale, ma non per LoRA BF16 e non per servizio + QLoRA simultanei.
3. **Se la priorità è LoRA/QLoRA serio locale**, una **RTX 6000 Ada 48GB** o L40S 48GB è più equilibrata della 5090 per memoria, ECC e stabilità, anche se meno veloce e più costosa.
4. **Configurazione raccomandata economicamente:** Strix/5090 locale per inference + **A100 80GB cloud** per il fine-tuning. Usare H100 solo per dataset CoT lunghi o molte iterazioni. Una sessione di 10 ore A100 al prezzo snapshot $2,79/h costa circa $27,90 di GPU [Δ][20], contro l'acquisto di una scheda professionale da migliaia di dollari/euro.
5. **Non acquistare 70B come prima macchina:** il beneficio sulla stessa famiglia Qwen2.5 è spesso 2–5 punti e non universale, mentre memoria, costo e latenza aumentano in modo drastico. Usarlo come riferimento cloud/teacher/evaluator.

### Piano operativo

- Fase 1: Qwen3-32B AWQ/GGUF Q4, contesto 8K, `/no_think` per dialogo e `/think` per problemi difficili.
- Fase 2: dataset con train/validation separati, 1–3 epoche, LoRA r=16 o QLoRA NF4 r=16; misurare un set privato italiano, coding e math.
- Fase 3: training iniziale su A100 80GB cloud; confrontare LoRA BF16 e QLoRA con stesso seed e stessa seq length.
- Fase 4: riportare l'adapter sulla macchina locale, verificare qualità su Q4 reale e solo dopo decidere se passare a rank 32, contesto maggiore o H100.
- Fase 5: tenere DeepSeek-R1-Distill-Qwen-32B e Qwen2.5-Coder-32B come **benchmark specialistici**, non come tre modelli da allenare contemporaneamente.

---

## Appendice A — Modelli affini 24–35B e generazione Qwen 3.5 / 3.6 / 3.8 (aggiornamento settembre 2026)

Questa appendice estende il perimetro su richiesta: include Qwen3.8-27B, Qwen3.6-27B, Qwen3.6-35B-A3B, Qwen3.5-27B/35B-A3B e altri rilasci recenti ~24–35B (Gemma 3 27B, GLM-4.7-Flash 30B-A3B, Mistral Small 3.1/3.2 24B). Non esiste un checkpoint denso ufficiale **Qwen3.8-32B** su Hugging Face Qwen: il denso comparabile di generazione 3.8 è **27B** (27,78B memorizzati) [21][22]. I punteggi qui sono quindi riportati **per famiglia/pesi reali**, in tabelle separate; non li fondo in un'unica media con i base model Qwen3 15-benchmark di §2.2.

### A.1 Perché non erano nel ranking principale

Il ranking §2–3 è ancorato a report/cards con protocolli identici (MMLU/MMLU-Pro/GPQA/GSM8K/MATH/HumanEval/MBPP ecc. con shot e decoding dichiarati) pubblicati entro maggio 2025 [1][4][5]. Qwen3.6 (aprile 2026) e Qwen3.8-27B (14 agosto 2026, 15:00 UTC, ModelScope/HF) [22] pubblicano invece, come evidenza primaria, **benchmark agentic e computer-use** (Terminal-Bench, SWE-bench Pro/DeepSWE, OSWorld, WebArena, RecreationBench) più HLE/GPQA/LiveCodeBench v6 [22][23]. Sono benchmark validi, ma **non omogenei** ai 15 base-benchmark Qwen3; mescolarli avrebbe violato la regola di comparabilità §1.3. Qui li espongo in tabelle dedicate con protocollo e caveat.

### A.2 Perimetro esteso — pesi, contesto, architettura, licenza

| Modello (HF ID) | Parametri | Architettura | Contesto nativo / esteso | BF16 checkpoint | Q4 GGUF indicativo | Licenza |
|---|---:|---|---|---|---:|---|
| Qwen3-32B [1][2] | 32,8B densi | 64 layer, GQA 64Q/8KV, head 128 | 32K nativo / 131K YaRN | 51,76 GiB (55,58 GB dec.) [12] → 62,75 GB con runtime [11] | ~19,1 GB AWQ / ~22–26 GB Q4_K_M | Apache 2.0 |
| Qwen3.8-27B [21][22] | **27,78B** densi (248.320 vocab) | 64 layer: 48×Gated DeltaNet + 16×GQA (24Q/4KV, head 256) + MTP | **262K** nativo / **1M YaRN (factor 4.0)** | **51,76 GiB / 55,58 GB dec.** BF16 [22] | **15,93 GiB** (+0,87 vision) Q4_K_M / **17,11 GB dec.** [22] | Apache 2.0 |
| Qwen3.6-27B [21] | ~27,8B densi | stesso schema ibrido 64 layer | 262K / ~1M | ~51–55 GB BF16 [E] | ~15–17 GB Q4 | Apache 2.0 |
| Qwen3.6-35B-A3B [21] | 35B tot / **3B attivi** MoE | 128 esperti, 8 attivi | 262K / ~1M | — | ~20–24 GB (dipende da formato) [E] | Apache 2.0 |
| Qwen3.5-27B [21] | 27B densi | dense transformer | 262K nativo | ~51 GB BF16 | ~15–17 GB Q4 | Apache 2.0 |
| Gemma 3 27B IT [24][25] | 25,6B non-emb (27B total con 417M vision) | GQA, 5:1 local/global (sw 1024) | **128K** (1B: 32K) | **54,0 GB** BF16 / **14,1 GB int4** / +KV 72,7 GB a 32K [24][S] | ~14–16 GB Q4 (terze parti) | Gemma license |
| GLM-4.7-Flash [26] | **30B tot / 3B attivi** MoE | MoE 128 esperti equiv. | **128K** | ~60 GB BF16 stimato [E] | ~14–16 GB Q4 [E] | MIT-ish (Zhipu) |
| Mistral Small 3.1/3.2 24B | 24B densi | dense | 128K (3.2: 128K) | ~48 GB BF16 | ~13–15 GB Q4 | Apache 2.0 |

Nota memoria: per Qwen3.8-27B il risparmio non è solo dai pesi, ma dalla **Gated DeltaNet** — solo 16 layer full-attention costruiscono KV cache classica. La stima vendor per quel componente è **65.536 byte/token** → **16 GiB a 262K, ~61 GiB a 1M** [22][E]; per Qwen3-32B la stima equivalente è 256 KiB/token → 8 GiB a 32K / 32 GiB a 128K [S][E]. “Corre su 24 GB” e “supporta 262K” non vanno fusi: la prima misura riguarda il file Q4, la seconda la capacità di contesto; collegarle senza includere KV/cache/overhead è scorretto [22].

### A.3 Benchmark ufficiali comparabili (Qwen-run, stesso harness)

La tabella più omogenea è quella del model card **Qwen3.6-27B** (HF README raw), che confronta con **Qwen3.5-27B, Qwen3.5-397B-A17B, Gemma4-31B, Claude 4.5 Opus e Qwen3.6-35B-A3B** sotto temperature e scaffold Qwen [21][S]. Estratto dei benchmark dove il confronto è 1:1 (valori %, [C] vendor, harness Claude Code per SWE/TBench):

**Coding agent / knowledge / reasoning (stessa harness Qwen):**

| Benchmark | Qwen3.5-27B | Qwen3.5-397B-A17B | Gemma4-31B | Qwen3.6-35B-A3B | **Qwen3.6-27B** | Qwen3.8-27B* |
|---|---:|---:|---:|---:|---:|---:|
| SWE-bench Verified | 75,0 | 76,2 | 52,0 | 73,4 | **77,2** | **61,7** [Pro]ᵃ |
| SWE-bench Pro | 51,2 | 50,9 | 35,7 | 49,5 | **53,5** | 61,7ᵃ |
| Terminal-Bench 2.0/2.1 | 41,6 | 52,5 | 42,9 | 51,5 | **59,3** | **73,0** |
| MMLU-Pro | 86,1 | 87,8 | 85,2 | 85,2 | **86,2** | —ᵇ |
| MMLU-Redux | 93,2 | 94,9 | 93,7 | 93,3 | **93,5** | —ᵇ |
| SuperGPQA | 65,6 | 70,4 | 65,7 | 64,7 | **66,0** | —ᵇ |
| GPQA Diamond | 85,5 | 88,4 | 84,3 | 86,0 | **87,8** | **89,2** |
| HLE (Humanity's Last Exam) | 24,3 | 28,7 | 19,5 | 21,4 | **24,0** | **30,8** |
| LiveCodeBench v6 | 80,7 | 83,6 | 80,0 | 80,4 | **83,9** | **90,3** |
| HMMT Feb 25 | 92,0 | 94,8 | 88,7 | 90,7 | **93,8** | — |
| AIME26 | 92,6 | 93,3 | 89,2 | 92,7 | **94,1** | — |

ᵃ *SWE-bench Pro di Qwen3.8-27B è sul set corretto 200K/Claude Code (≠ SWE-bench Verified); non confrontarlo 1:1 con SWE-bench Verified 75,0/77,2 senza nota.* [22][23]  
ᵇ *Qwen3.8-27B non pubblica MMLU-Pro/Redux/SuperGPQA nella tabella launch; i suoi MMLU-Pro sono riportati solo da terze parti non omogenee → qui **N.D.*** [22]  
Fonte: Qwen3.6-27B README benchmark table [21][S]; Qwen3.8-27B righe da model card/analisi Kingy/Qubrid [22][23][C].

**Agent / computer-use (solo Qwen3.8-27B card, vendor-run, cfr. Qwen3.6):** [22][23][C]

| Benchmark | Qwen3.6-27B | **Qwen3.8-27B** | Δ [Δ] | Nota |
|---|---:|---:|---:|---|
| OSWorld-Verified | 63,9 | **84,3** | **+20,4** | desktop automation |
| WebArena-Verified | 48,8 | **64,8** | **+16,0** | browser automation |
| AndroidWorld | 70,3 | **81,9** | +11,6 | mobile |
| Vision2Web | 45,0 | **62,9** | +17,9 | frontend da screenshot (Claude Code + GPT-5.4 judge) |
| SWE-MM (multimodal) | 25,7 | **38,6** | +12,9 | dev split modificato |
| DeepSWE 1.1 | 13,3 | **42,2** | **+28,9** | |
| QwenSWEBench (interno) | 49,3 | **79,0** | +29,7 | avg3, interno Qwen — non riproducibile esterno |
| CoWorkBench (interno) | 61,0 | **70,7** | +9,7 | interno 1h timeout |
| IFBench | 69,1 | **79,5** | +10,4 | instruction following |
| GPQA Diamond | 87,8 | **89,2** | +1,4 | |
| Humanity's Last Exam | 24,0 | **30,8** | +6,8 | judge GPT-4o |
| LiveCodeBench v6 | 83,9 | **90,3** | +6,4 | |

Contro Qwen3.7-Plus, Qwen3.8-27B vince **10/12** righe testo e perde GPQA Diamond e HLE; contro Muse Glimmer-30B vince tutte le righe sovrapposte [22][C]. I caveat vendor sono sostanziali: task set corretto, judge GPT-4o per HLE, Vision2Web giudicato da GPT-5.4, NL2Repo senza comandi di rete, MathVision con prompt asimmetrico [22][23][P].

**Verifica indipendente (Artificial Analysis):** Intelligence Index (9 eval: GDPval-AA v2, T³-Banking, T-Bench v2.1, SciCode, HLE, GPQA Diamond, CritPt, AA-Omniscience, AA-LCR) — **Qwen3.8-27B xhigh 52**, medium 44, non-reasoning 35; **Qwen3.6-27B reasoning 38** [23][M]. Differenza **+14 punti a parità di 27,78B e architettura quasi identica** → guadagno da post-training/RL/distillazione, non da scala [22][23]. Agentic Index: 50,877 (≈51), sopra Claude Opus 4.8 max di <1 punto (margine stretto, varianti effort diverse) [23][M]. **Verbosity:** xhigh produce ~3,3× la mediana peer (160M output token nel benchmark) e casi di 22K reasoning token per 3,2K output: i punteggi xhigh non sono gratuiti in latenza/costo [23][P].

### A.4 Benchmark classici standard (Gemma 3 27B IT) — non mischiabili con gli agentic

Dalla Gemma 3 Technical Report / model card [24][25][S][C]:

| Modello | MMLU-Pro | GPQA Diamond | MATH | HumanEval | GSM8K | LiveCodeBench | IFEval |
|---|---:|---:|---:|---:|---:|---:|---:|
| Gemma 3 27B IT [24][25] | **67,5** | **42,4** | **89,0** | **87,8** | **95,9** | **29,7** | **90,4** |
| Gemma 3 27B PT — MMLU-Pro COT 52,2; MMLU 78,6 [24][25][S] | | | | | | | |
| Qwen3-32B Base (cfr. §2.2) | 65,54 | 49,49 | 61,62 | — | 93,40 | — | — | 
| Qwen3-32B thinking (card) | 90,9 (MMLU-Redux) | 68,4 | — | — | — | — | — |

Gemma 3 27B IT eccelle in istruzioni (IFEval 90,4) e ha HumanEval 87,8 competitivo, ma il suo GPQA Diamond (42,4) e MMLU-Pro (67,5) non vanno confrontati 1:1 con gli SWE-bench/Official harness Qwen3.8: famiglie diverse, shot e scaffold diversi. Per l'utente, Gemma 3 27B resta un'alternativa Apache 2.0 solida quando priorità è **istruzione + multimodal leggero (SigLIP) + 128K**, non coding agentic di Qwen3.8.

**Altri vicini 24–30B recenti (per completezza, tabella non omogenea — uso [P]):**

| Modello | Nota | Evidenza utile |
|---|---|
| GLM-4.7-Flash 30B-A3B [26] | 30B/3B MoE, 128K | Vendor: AIME 2025 ~91,6%, GPQA ~75,2%, SWE-Bench Verified ~59,2% (harness GLM, non Qwen) — segnali forti su agent/math, ma non direttamente comparabili ai numeri Qwen3.8 senza rerun [P] |
| Mistral Small 3.1/3.2 24B | 24B dense, 128K | Aggiornamento 3.2 focalizzato su repetition/instruction; la 3.1 migliora su GPQA/HumanEval/MMLU-Pro vs Small 3 (dettagli model card) [P] |
| Phi-4 14B/14B-Reasoning | 14B dense | MMLU ~80–84, GPQA ~56, MATH 80+ (Microsoft): più piccolo della classe, utile come efficient frontier per confronto costo, non come sostituto 30B [P] |

### A.5 Confronto inter-generazionale sintetico

| Generazione | MMLU-Pro (PT/IT) | GPQA Diamond | HLE | OSWorld | SWE-bench (best) | Nota |
|---|---:|---:|---:|---:|---:|---|
| Qwen3-32B Base (§2.2) | 65,54 | 49,49 | — | — | — | base model 15-bench |
| Qwen3.5-27B | 86,1 | 85,5 | 24,3 | — | 75,0 (Verified) | card Qwen3.6 [21] |
| Qwen3.6-27B | **86,2** | **87,8** | **24,0** | 63,9 | **77,2** | +1–2 pt su Qwen3.5 |
| **Qwen3.8-27B** | N.D. (non pubblicato omogeneo) | **89,2** | **30,8** | **84,3** | **61,7 Pro / 79,0 interno** | **+14 su AA Index vs 3.6 a parità di architettura** [23] |
| Gemma 3 27B IT | 67,5 | 42,4 | — | — | — | standalone |

Lettura: **a pari decoder (3.6→3.8) il salto è di post-training**; il vantaggio 3.8 si concentra su **agentic/computer-use (+20 OSWorld, +16 WebArena, +28 DeepSWE)** più che su conoscenza enciclopedica (MMLU-Pro/HLE restano terreno di frontier >100B) [22][23].

### A.6 Hardware per inferenza — nuovi modelli 27–35B

Tutti i 27B densi hanno checkpoint BF16 ~51–55 GiB; i MoE 30–35B-A3B hanno file più piccoli ma KV/cache per sequenza analoga. Le velocità sotto sono [E] salvo Kingy lab dove citato.

| Modello | Q4 file | VRAM inferenza (Q4, contesto corto 4–8K) | VRAM inferenza 262K (BF16 KV pieno) | Decode [E] | Consumo |
|---|---:|---:|---:|---:|---:|
| Qwen3.8-27B | 15,93 GiB (+0,87) / 17,11 GB dec [22] | **24 GB possibile** a 4-bit contesto moderato [22][M] | 51,76 GiB pesi + 16 GiB KV = **67,76 GiB** min [22][E] — non 24 GB | 20–38 tok/s su 5090; 10–20 Strix Halo; 35–60 SGLang H100 [E] | 5090 575W; Strix 80–140W |
| Qwen3.6-27B | ~15–17 GB | 24 GB possibile | ~68 GiB con 262K [E] | simile a 3.8 (±10%) [E] | idem |
| Qwen3.6-35B-A3B | ~20–24 GB | 24 GB possibile | ~48 GiB pesi equiv. + KV 16 GiB → **~64 GiB** [E] | solo 3B attivi → **1,5–2×** più veloce del dense 27B a pari GPU [E] | <500W sistema 5090 |
| Qwen3.5-27B | ~15–17 GB | 24 GB possibile | ~68 GiB | simile | idem |
| Gemma 3 27B IT | ~14–16 GB Q4 | 24 GB possibile; full BF16+KV 72,7 GB a 32K [24][S] | 128K nativo: il local/global 5:1 riduce KV vs Qwen → più efficiente su finestre lunghe [24][E] | 18–35 tok/s 5090 [E] | 0,6–0,9 kW |
| GLM-4.7-Flash 30B-A3B | ~14–16 GB | **22–26 GB** consigliati | MoE 3B attivi → contesto lungo più leggero | 25–45 tok/s 5090 [E] | 300–575W |

Nota: il FP8 ufficiale Qwen3.8 (28,76 GiB) **non entra in 24 GB** con 262K nativo (28,76+16=44,76 GiB) e lascia poco margine anche su 48 GB senza ottimizzazioni cache [22]. Con Q4 + KV FP8/INT8 e contesto 64–128K è realistico; per 262K–1M servono 48–80 GB o YaRN parziale.

### A.7 LoRA/QLoRA sui nuovi 27–30B — ricalcolo

| Classe | BF16 pesi | QLoRA r=16 NF4 (seq 512–1K) | QLoRA con 262K o CoT lungo | LoRA BF16 |
|---|---:|---:|---:|---:|
| 27–28B dense (3.6/3.8/3.5) | ~55 GB [E] | **24–32 GB** molto ottimizzato; **32–40 GB raccomandati** [E] | 40–64 GB [E] | ~70–90 GB (96 GB ideale) [E] |
| 32,8B dense (Qwen3-32B) | ~65,6 GB [E] | 28–32 GB opt; 40–48 GB racc. [E] | 40–48 GB (2K) / 80 GB (4–8K) [E] | 80–96 GB [E] |
| 30–35B-A3B MoE | ~60–70 GB tot (3B attivi) [E] | **22–32 GB** (solo layer attivi + cache minore) [E] | 32–48 GB [E] | 80–96 GB [E] |

Conseguenza: **Qwen3.8-27B QLoRA entra più facilmente su 24 GB rispetto a Qwen3-32B**, ma solo a **seq corta/mediana** e micro-batch 1; per finetuning su documenti lunghi (262K) la KV e gli hidden state lineari annullano il vantaggio. Per MoE 35B-A3B, il training tocca 3B attivi ma richiede comunque gli optimizer state del base → non considerarlo “3B” per dimensionamento.

### A.8 Quando scegliere 27B vs 32B vs MoE (sintesi estesa)

| Obiettivo | Primo | Perché | Hardware |
|---|---:|---|---|
| **Singolo modello general-purpose locale** | **Qwen3.8-27B** | Miglior profilo agentic + reasoning recente a parità di ~27–28B; Apache 2.0; 262K nativo; +14 AA Index su 3.6 [22][23] | 24 GB Q4 moderato; 48 GB per 262K; A100 40GB QLoRA |
| **Massima qualità standard (MMLU/MMLU-Pro base homog.)** | Qwen3-32B Base (§2.2) | Unico con 15-bench omogenei fino a maggio 2025; ancora riferimento rigoroso per §4 | Strix 128GB / 5090 32GB |
| **Coding agent di frontiera su 30B** | **Qwen3.8-27B > Qwen3.6-27B > Qwen3.5-27B** | 73,0 T-Bench, 61,7 SWE-Pro, 90,3 LCB v6; progressione documentata [21][22] | Strix QLoRA stretto o A100 80 |
| **Throughput / multi-utente** | **Qwen3.6-35B-A3B** | 3B attivi → ~1,5–2× tok/s vs dense 27B [E]; buon compromesso costo/latenza | 24–48 GB |
| **Istruzioni pure / IFEval / multimodal leggero** | **Gemma 3 27B IT** | IFEval 90,4, HumanEval 87,8, SigLIP + 128K efficiente [24][25] | 24 GB Q4 |
| **Budget 24 GB rigido** | Gemma 3 27B / Mistral Small 24B / GLM-4.7-Flash | File più piccoli o MoE attivo ridotto; meno qualità agentic di Qwen3.8 [E][P] | 24 GB |

**Raccomandazione aggiornata per l'utente (priorità 32B + LoRA):** se l'uso è **coding agent / computer-use / automazione documentale**, **spostare il default da Qwen3-32B a Qwen3.8-27B** come modello primario locale (mantieni Qwen3-32B come fallback rigoroso finché non hai una tua misura MMLU-Pro/GPQA sul tuo harness). Se l'uso è **conoscenza/italiano generale con budget 32B**, resta su Qwen3-32B (§2–3) finché Qwen non pubblica per 3.8 una tabella MMLU-Pro/GPQA con shot identici a §1.2.

### A.9 Limiti e cosa resta non verificabile

- MMLU-Pro/GPQA/SuperGPQA di Qwen3.8-27B con **stesso shot della Tabella 4 Qwen3** → **N.D.**: non pubblicato in forma omogenea [22].
- Token count, dataset, knowledge cutoff, RL envs, distillation recipe di Qwen3.8-27B → **non divulgati** [22].
- FP8 “quasi identico a BF16” dichiarato per Qwen3.8 → vendor claim fino a rerun 1:1 [22][P].
- Quantizzazioni GGUF terze parti (Unsloth dynamic etc.) → **non ufficiali**; qualità agentic/vision può degradare vs BF16 [22].
- YaRN 262K→1M: static YaRN penalizza prompt corti; usare factor 2.0 per ~500K, non 4.0 di default se il tipico è 8K [22][E].
- Tok/s: dipendono da engine (Transformers/vLLM/SGLang/llama.cpp), precisione KV, offload, temperatura 1.0 vs 0,6 e reasoning effort (xhigh >> medium) [22][23][E].

---

## Fonti numerate

1. **Qwen Team, “Qwen3 Technical Report”, arXiv 2505.09388**, tabelle base model, protocolli e confronto 32B/72B: https://arxiv.org/html/2505.09388v1
2. **Qwen, model card Qwen3-32B-AWQ**, performance thinking/non-thinking, configurazione, quantizzazione e best practice: https://huggingface.co/Qwen/Qwen3-32B-AWQ/raw/main/README.md
3. **DeepSeek AI, model card DeepSeek-R1-Distill-Qwen-32B**, base model e licenza del checkpoint: https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B/raw/main/README.md
4. **DeepSeek-AI, “DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning”, arXiv 2501.12948**, tabelle R1 e distillati 32B/70B: https://arxiv.org/html/2501.12948v1
5. **Qwen, “Qwen2.5-LLM: Extending the boundary of LLMs”**, benchmark base/instruct di Qwen2.5-32B e Qwen2.5-72B: https://qwenlm.github.io/blog/qwen2.5-llm/
6. **Qwen Team, “Qwen2.5-Coder Technical Report”, arXiv 2409.12186**, architettura e benchmark coding: https://arxiv.org/html/2409.12186v3 — i punteggi ufficiali HumanEval/MBPP/LiveCodeBench/Aider della serie sono annunciati anche nel blog Qwen: https://qwenlm.github.io/blog/qwen2.5-coder-family/
7. **Meta/Llama 3.3 70B Instruct model card**, architettura, licenza e link agli evaluation details: https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct
8. **Meta, dataset/evaluation details Llama 3.1 70B Instruct**, riferimento ai protocolli Meta: https://huggingface.co/datasets/meta-llama/Llama-3.1-70B-Instruct-evals
9. **Mistral AI, “Large Enough — Mistral Large 2”**, benchmark dichiarati per il modello 123B: https://mistral.ai/news/mistral-large-2407/
10. **SemiAnalysis InferenceX, Llama 3.3 70B**, aggregazione di punteggi riportati e confronto MMLU/IFEval: https://inferencex.semianalysis.com/model/llama-3-3-70b
11. **Qwen documentation, “Speed Benchmark”**, H20, footprint e throughput Qwen3 BF16/FP8/AWQ/SGLang: https://qwen.readthedocs.io/en/latest/getting_started/speed_benchmark.html
12. **NVIDIA, GeForce RTX 5090**, 32GB, TGP 575W, PSU e MSRP: https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/
13. **NVIDIA, RTX 6000 Ada Generation**, 48GB ECC, 960GB/s e 300W: https://www.nvidia.com/en-us/products/workstations/rtx-6000/
14. **AMD, Ryzen AI Max+ 395**, memoria massima 128GB LPDDR5x-8000 e specifiche: https://www.amd.com/en/products/processors/laptop/ryzen/ai-300-series/amd-ryzen-ai-max-plus-395.html
15. **GMKtec Europe, EVO-X2 Ryzen AI Max+ 395**, snapshot prezzo/configurazione 128GB: https://de.gmktec.com/en/products/gmktec-evo-x2-amd-ryzen%E2%84%A2-ai-max-395-mini-pc-1
16. **Apple, Mac Studio technical specs/newsroom**, configurazione M4 Max fino a 128GB unified memory: https://support.apple.com/en-us/122211 e https://www.apple.com/newsroom/2025/03/apple-unveils-new-mac-studio-the-most-powerful-mac-ever/
17. **Hu et al., “LoRA: Low-Rank Adaptation of Large Language Models”, arXiv 2106.09685**, metodo e riduzione dei parametri: https://arxiv.org/abs/2106.09685
18. **Dettmers et al., “QLoRA: Efficient Finetuning of Quantized LLMs”, arXiv 2305.14314**, NF4, double quantization, paged optimizer e 65B su 48GB: https://arxiv.org/abs/2305.14314
19. **Hugging Face Transformers, “Parameter-efficient fine-tuning”**, PEFT/LoRA/QLoRA, adapter e device map: https://huggingface.co/docs/transformers/en/peft
20. **Lambda AI, GPU cloud pricing**, snapshot A100/H100/A6000/RTX 6000 e capacità VRAM: https://lambda.ai/pricing
21. **Qwen, Qwen3.6-27B model card / HF README raw** (tabella benchmark Qwen3.5/3.6/3.6-35B, 21 apr 2026; RAW): https://huggingface.co/Qwen/Qwen3.6-27B/raw/main/README.md
22. **Qwen/Qwen3.8-27B — Kingy.ai evidence-led guide** (14–22 ago 2026, con HF config, ModelScope timestamp, BF16 55,58 GB / FP8 30,88 GB, Q4 15,93 GiB, KV 65.536 byte/token → 16 GiB a 262K, tabelle launch e caveat harness/judge): https://kingy.ai/blog/qwen3-8-27b-specs-benchmarks-local-hardware/ — model cards Qwen: https://huggingface.co/Qwen/Qwen3.8-27B e https://huggingface.co/Qwen/Qwen3-32B
23. **Qubrid AI, “Qwen3.8-27B Benchmarks: Official and Independent Results”** (31 ago 2026, raccolta tabelle Qwen + Artificial Analysis Intelligence/Agentic Index 52/51, ExtractBench, caveat harness): https://www.qubrid.com/blog/qwen38-27b-benchmarks-official-and-independent-results
24. **Gemma 3 Technical Report, arXiv 2503.19786** (architettura 5:1 local/global, 128K, memoria BF16/ int4 / +KV a 32K): https://arxiv.org/html/2503.19786v1
25. **Google, Gemma 3 model card** (14T token aug 2024 cutoff, tabelle PT/IT MMLU-Pro 52,2→67,5, HumanEval, GSM8K, IFEval): https://ai.google.dev/gemma/docs/core/model_card_3
26. **Zhipu AI, GLM-4.7 / GLM-4.7-Flash** (annuncio 30B-A3B, 128K, benchmark AIME/GPQA/SWE-Bench): https://z.ai/blog/glm-4.7 e https://www.marktechpost.com/2026/01/20/zhipu-ai-releases-glm-4-7-flash-a-30b-a3b-moe-model-for-efficient-local-coding-and-agents/

### Limiti residui da verificare prima dell'acquisto

- tok/s locali su Strix Halo, Apple, 5090 e RTX6000 per **questo specifico GGUF/AWQ**: richiedono una misura sul modello, backend e contesto dell'utente;
- VRAM di training con la versione esatta di Unsloth/PEFT/Flash-Attention: una modifica di kernel può spostare la soglia di alcuni GB;
- disponibilità e prezzo italiano effettivo di GPU cloud, VAT, storage e quota minima;
- qualità del fine-tuned adapter sul set privato: nessun benchmark standard sostituisce la validazione sul caso d'uso.
