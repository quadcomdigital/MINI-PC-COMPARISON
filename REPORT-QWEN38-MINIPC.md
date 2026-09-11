# Verifica indipendente: Qwen3.8-27B su mini PC piccolo ed economico

**Data:** 10 settembre 2026
**Committente:** progetto "stack AI locale", iterazione dopo la verifica dual-Tesla P40
**Riferimenti:** i 5 report precedenti (`REPORT-VERIFICA-INDIPENDENTE.md`, `ANALISI-RIGOROSA.md`, `ANALISI-MINIPC-COMPONENTI.md`, `VARIANTI-70B.md`, `RANKING-MODELLI-FINETUNING.md` + Appendice A). Nessuno di questi è modificato.
**Domanda:** abbandonare esplicitamente l'approccio "70B su hardware grande" e verificare se **Qwen3.8-27B** (generazione Qwen 3.5/3.6/3.8, rilasciato 14 ago 2026) gira su una macchina **più piccola e meno costosa** del progetto P40, restando dentro il caso d'uso e la soglia di qualità adottata in tutti i report.

> **Nota metodologica.** Questo documento è una verifica indipendente, non un riassunto dei report precedenti. Ogni dato chiave è classificato con la legenda sotto; i benchmark agentic/computer-use di Qwen3.8 (Terminal-Bench, SWE-bench Pro, OSWorld, WebArena) sono trattati in tabelle **separate** dai benchmark classici, perché usano harness diversi e non sono omogenei. Le stime di throughput sono [E] finché non esiste una misura sul file esatto: nessuna viene venduta come promessa. I prezzi sono snapshot di annunci/listini e non quotazioni vincolanti.

**Legenda dell'evidenza (usata in tutto il documento):**

- **[S] Specifica:** dato dichiarato da produttore/documentazione primaria.
- **[M] Misura:** benchmark pubblicato con hardware, modello e condizioni leggibili.
- **[C] Claim/vendor:** risultato ufficiale dichiarato dal vendor, non verifica indipendente.
- **[P] Proxy/stima personale:** misura reale ma con modello/quant/hardware non identici, oppure ipotesi.
- **[E] Proiezione di progetto:** intervallo progettuale usato per decidere in assenza di misura; va validato.
- **[€] Prezzo:** prezzo osservato (annuncio, listino, marketplace); non è una fattura né un listino garantito.
- **[Δ] Differenza calcolata** in questo documento da numeri citati.

**Soglia di progetto "chat fluida"** (adottata in tutti i report precedenti e qui):

- decode **≥ 8 tok/s** a concorrenza 1;
- **TTFT caldo ≤ 2 s** su prompt di 512 token (≤ 5 s a 4K);
- 3–4 tok/s = **sperimentale**, non soluzione primaria.

---

## 1. Sintesi esecutiva e verdetto

### 1.1 Risposte alle sei domande del task

**1. Ha senso rimpiazzare il progetto 70B/P40 con Qwen3.8-27B su mini PC? → SÌ, con un candidato hardware specifico e una condizione.**

La combinazione Qwen3.8-27B + mini PC compatta risolve i tre difetti fatali della dual-P40 identificati dalla verifica indipendente (550–750 W alla presa, 3–6 tok/s [E], software datacenter del 2016 in uscita) e abbassa drasticamente tutti i costi. Il passo qualità-caso d'uso è documentato: Qwen3.8-27B supera Qwen3.6-27B (stesso decoder, stessa architettura ibrida a 64 layer) di **+14 punti sull'Artificial Analysis Intelligence Index** (52 xhigh vs 38, stessa base da 27,78B → il guadagno è post-training/RL, non scala) e porta OSWorld-Verified da 63,9 a **84,3** (+20,4), WebArena da 48,8 a **64,8** (+16,0), DeepSWE da 13,3 a **42,2** (+28,9) [C/M, tabelle agentic, sezione 2]. L'A.8 del report Ranking ha già dichiarato: se l'uso è coding agent/computer-use/automazione, **il default passa da Qwen3-32B a Qwen3.8-27B**. Il caso d'uso reale del progetto (assistente/agent locale, non chat enciclopedica) è esattamente il profilo in cui il vantaggio 3.8 si concentra.

**La condizione:** il mini PC deve appartenere alla classe con banda di memoria adeguata (Strix Halo 128 GB / Apple Silicon con memoria unificata / host + GPU discreta). I mini PC dual-channel DDR5 (780M/890M, 96 GB) restano sotto la soglia chat anche per un 27B Q4, perché il limite è la **banda** (89,6 GB/s), non la capacità: vedi sezione 3.

**2. Qualità sufficiente per il caso d'uso? → SÌ per l'uso agentic/assistente; con una debolezza dichiarata e non colmabile con fonti terze.**

- Punti di forza misurati e rilevanti per un agent locale: Terminal-Bench 2.0/2.1 **73,0** (vs 59,3 del 3.6), LiveCodeBench v6 **90,3**, GPQA Diamond **89,2**, HLE **30,8**, Agentic Index ≈ **51** (sopra Claude Opus 4.8 di meno di 1 punto, margine stretto tra varianti effort diverse) [C/M].
- Debolezza: **MMLU-Pro N.D.** — la card non pubblica MMLU-Pro/MMLU-Redux/SuperGPQA omogenei; le uniche cifre circolanti sono di terze parti non omogenee e **non vengono usate qui per colmare il buco**. Il gap enciclopedico vs un 70B esiste ed è non quantificabile onestamente oggi.
- Caveat operativo reale: in modalità `xhigh` produce **~3,3× la mediana dei peer** in output token (160M token nel benchmark; casi con 22K reasoning token per 3,2K di output) [P] → su hardware locale la modalità reasoning va governata (medium/non-reasoning per chat, xhigh solo per task difficili), altrimenti la verbosity mangia la latenza.

**3. Hardware minimo per la soglia "chat fluida"? → La classe minima credibile è ~256 GB/s di banda con il modello interamente in memoria; la fascia d'ingresso credibile è ~1.000–1.700 €.**

Il calcolo chiave (sezione 3): con Q4_K_M da **15,93 GiB** (pesi), per superare 8 tok/s di decode serve leggere ~17,1 GB per token generato → banda *effettiva* minima ≈ **137 GB/s** a 8 tok/s, e in pratica **170–230 GB/s di picco dichiarato** per compensare l'efficienza reale (60–80%). Con margine KV/sistema: **≥ 32 GB di memoria** (comfort 64 GB), contesti 4–8K comodi, contesti 32K+ ok, 262K pieni fuori portata per la classe mini (vedi sezione 7). Strix Halo (256 GB/s) e Apple M4 Pro (273 GB/s) sono le prime classi che superano la barra con margine; tutto ciò che sta a 89–120 GB/s no. Fascia di ingresso credibile: **~1.000–1.700 €** (host + GPU discreta usata); fascia comoda: **1.200–1.900 €** (Apple usato/mini nuovo).

**4. Varianti component-driven e marche sconosciute:** censite in sezione 4. Il criterio è la combinazione CPU/banda + iGPU/dGPU + quantizzazione, non il marchio. Il censimento conferma che la fascia CN (Taobao/1688) ha i prezzi migliori proprio sulle combinazioni vincenti (395/128 GB), con la serie di caveat d'importazione già formalizzati in VARIANTI-70B.md.

**5. Confronto economico vs P40 e vs cloud:** la sostituzione **vince ovunque**. Contro la dual-P40 il mini PC risparmia ~78–126 €/mese di elettricità da solo: il differenziale d'acquisto si ripaga in **6–13 mesi** per i candidati economici (Mac, host+GPU), **~24 mesi** per lo Strix Halo importato CN e **~3,3 anni** per il canale EU a prezzo pieno (sezione 5.4). Contro il cloud on-demand sullo stesso modello, il locale vince sopra ~100–160 h/mese di uso; sotto, vince il cloud (sezione 5.5).

**6. Verdetto finale:** scartare definitivamente la dual-P40 (già declassata dai report precedenti) **e** non costruire nulla di nuovo per il 70B. Configurazione raccomandata: **mini PC Ryzen AI Max+ 395 con 128 GB UMA** (o Apple Silicon equivalente se si accetta ecosistema chiuso) che esegue Qwen3.8-27B Q4 a soglia da verificare, con test di accettazione obbligatorio prima dell'acquisto definitivo (sezione 8). Fallback economico: host 96 GB + GPU discreta usata; scenario "mini economicissimo": solo se il contesto d'uso si restringe a 4–8K e si accetta una CPU-only sperimentale.

### 1.2 Verdetto in una riga

**Sì alla sostituzione, con Qwen3.8-27B come modello primario locale, ma solo su hardware con ≥256 GB/s di banda (Strix Halo 128 GB o equivalente Apple); spendendo 1.000–1.900 € per i candidati economici (o 2.300–2.700 € per lo Strix importato) invece di 850–1.250 € + 78–126 €/mese di elettricità, con TTFT/decode da certificare con il protocollo di sezione 8; la dual-P40 viene demolita anche economicamente.**

---

## 2. Il modello: Qwen3.8-27B, generazione 3.5/3.6/3.8

### 2.1 Cosa è (dati già verificati, riportati senza reinventarli)

Fonte primaria: model card HF/ModelScope e analisi Kingy/Qubrid già censite in Appendice A di `RANKING-MODELLI-FINETUNING.md` [21][22][23] lì.

| Proprietà | Valore | Evidenza |
|---|---|---|
| Parametri | **27,78B densi** (vocab 248.320) | [S][22] |
| Architettura | 64 layer: **48×Gated DeltaNet + 16×GQA** (24Q/4KV, head 256) + MTP | [S][22] |
| Contesto | **262K nativo / 1M YaRN (factor 4.0)** | [S][22] |
| Checkpoint BF16 | **51,76 GiB** (55,58 GB decimali) | [S][22] |
| FP8 | 28,76 GiB | [S][22] |
| **Q4_K_M GGUF** | **15,93 GiB** (+0,87 vision) → **17,11 GB decimali** | [S][22] |
| KV cache | **65.536 byte/token** → **16 GiB @262K**, ~61 GiB @1M | [S/E][22] |
| Licenza | **Apache 2.0** | [S][22] |
| Rilascio | 14 ago 2026, 15:00 UTC | [S][22] |

L'architettura ibrida è il fatto tecnico decisivo per il sizing: **solo 16 layer su 64 sono full-attention** che costruiscono KV cache classica; gli altri 48 sono Gated DeltaNet con stato ricorrente lineare. Per questo la KV cache è 65,5 KB/token invece dei ~256 KB/token di un dense transformer classico della stessa classe (Qwen3-32B). Conseguenza pratica: un contesto 32K costa **~2 GiB** di cache invece di ~8 GiB, e la promessa "262K" non richiede i 67,76 GiB di un full-attention 27B a quel contesto — ma **resta comunque fuori dalla classe mini**: 15,93 GiB pesi + 16 GiB KV @262K = **~32 GiB solo per modello+cache**, prima di sistema, runtime e di ogni altra cosa [Δ/S][22].

### 2.2 Benchmark: due tabelle, non una (regola anti-mescolamento)

**Tabella A — Benchmark agentic/computer-use, harness vendor (Claude Code, judge GPT-4o/GPT-5.4), solo Qwen3.8-27B card vs Qwen3.6-27B [C][22][23]:**

| Benchmark | Qwen3.6-27B | **Qwen3.8-27B** | Δ | Nota |
|---|---:|---:|---:|---|
| OSWorld-Verified | 63,9 | **84,3** | +20,4 | automazione desktop |
| WebArena-Verified | 48,8 | **64,8** | +16,0 | automazione browser |
| AndroidWorld | 70,3 | **81,9** | +11,6 | mobile |
| Vision2Web | 45,0 | **62,9** | +17,9 | frontend da screenshot (judge GPT-5.4) |
| SWE-MM (multimodal) | 25,7 | **38,6** | +12,9 | dev split modificato |
| DeepSWE 1.1 | 13,3 | **42,2** | +28,9 | |
| QwenSWEBench (interno Qwen) | 49,3 | **79,0** | +29,7 | **non riproducibile esternamente** |
| CoWorkBench (interno) | 61,0 | **70,7** | +9,7 | interno, timeout 1h |
| IFBench | 69,1 | **79,5** | +10,4 | instruction following |
| Terminal-Bench 2.0/2.1 | 59,3 | **73,0** | +13,7 | harness Claude Code |
| SWE-bench Pro | 53,5 | **61,7** | +8,2 | set corretto 200K ≠ SWE-bench Verified: **non confrontare 1:1** con i 77,2/75,0 di 3.6/3.5 |
| GPQA Diamond | 87,8 | **89,2** | +1,4 | |
| HLE | 24,0 | **30,8** | +6,8 | judge GPT-4o |

Caveat vendor sostanziali [C/P][22][23]: task set corretti dal vendor, judge GPT-4o/GPT-5.4, NL2Repo senza comandi di rete, MathVision con prompt asimmetrico. Questi numeri sono **dichiarazioni ufficiali**, non repliche indipendenti.

**Tabella B — Verifica indipendente Artificial Analysis (9 eval: GDPval-AA v2, T³-Banking, T-Bench v2.1, SciCode, HLE, GPQA Diamond, CritPt, AA-Omniscience, AA-LCR) [M][23]:**

| Modello | Intelligence Index | Note |
|---|---:|---|
| Qwen3.8-27B xhigh | **52** | ~3,3× la mediana dei peer in output token [P] |
| Qwen3.8-27B medium | 44 | compromesso qualità/latenza |
| Qwen3.8-27B non-reasoning | 35 | chat veloce |
| Qwen3.6-27B reasoning | 38 | |

**Δ = +14 punti a parità di 27,78B e architettura quasi identica** → il guadagno è post-training/RL/distillazione, non scala. Agentic Index: 50,877 ≈ 51, sopra Claude Opus 4.8 max di meno di 1 punto (margine stretto, varianti effort diverse) [M][23].

**Cosa NON è nella tabella (e non va colmato):** MMLU-Pro, MMLU-Redux, SuperGPQA di Qwen3.8-27B **N.D.** — non pubblicati in forma omogenea nella launch card. Il report Ranking ha correttamente rifiutato di usare le cifre di terze parti non omogenee. Anche qui: **non le uso**. Consequenza per il giudizio: la qualità "enciclopedica" del 3.8 rispetto a un 70B è **non quantificabile oggi**; si può dire solo che GPQA Diamond 89,2 e HLE 30,8 (con caveat judge) sono valori alti per la classe.

### 2.3 Confronto inter-generazionale (riassunto, da Appendice A)

| Generazione | MMLU-Pro | GPQA Diamond | HLE | OSWorld | SWE-bench best | AA Index |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-32B Base | 65,54 | 49,49 | — | — | — | — |
| Qwen3.5-27B | 86,1 | 85,5 | 24,3 | — | 75,0 (Verified) | — |
| Qwen3.6-27B | 86,2 | 87,8 | 24,0 | 63,9 | 77,2 (Verified) | 38 (reasoning) |
| **Qwen3.8-27B** | **N.D.** | **89,2** | **30,8** | **84,3** | 61,7 Pro / 79,0 interno | **52 (xhigh)** |

Lettura onesta: **il salto 3.6→3.8 è di post-training**, e il vantaggio si concentra su agentic/computer-use. La progressione di Qwen rispetto a Gemma 3 27B IT / GLM-4.7-Flash / Mistral Small 24B è quella documentata in A.4/A.8: Qwen3.8 vince il profilo "coding agent/computer-use"; Gemma 3 27B resta il riferimento per istruzioni pure/IFEval 90,4/multimodal leggero; GLM-4.7-Flash per MoE throughput.

### 2.4 Il profilo va bene per il caso d'uso? (giudizio esplicito)

Il caso d'uso reale del progetto (dichiarato nei report precedenti) **non è** chat enciclopedica ma **assistente/agent locale**: automazione desktop/browser, coding agent, sessioni lunghe con contesto residente. Per questo profilo:

1. **Il vantaggio 3.8 è esattamente nel punto giusto.** OSWorld +20,4 e WebArena +16,0 sono i benchmark che assomigliano di più al lavoro reale di un agent locale. Il salto +14 AA Index con architettura identica indica che anche i task nuovi del 3.8 rispetto al 3.6 (stesso decoder) sono maggiormente gestibili, non solo più copiati dal benchmark.
2. **Il 262K nativo con KV lineare è un vantaggio operativo reale** per un assistente che tiene una sessione residente: a 32K la cache costa ~2 GiB [Δ], quindi su una macchina da 128 GB UMA si può tenere il modello + contesti molto lunghi + altro.
3. **MMLU-Pro N.D. è un rischio accettabile ma reale.** Per l'uso agentic non è la metrica decisiva; per risposte enciclopediche in italiano il 3.8 potrebbe reggere peggio di un Qwen2.5-72B — e non posso dimostrare il contrario. Mitigazione: mantenere un secondo modello (Qwen3-32B o Gemma 3 27B) per quel profilo, oppure usare il cloud per i task di conoscenza pura.
4. **La verbosity xhigh è il vero rischio operativo su hardware mini.** A 3,3× la mediana dei peer in token, un reasoning xhigh triplica il tempo di risposta percepito e l'energia per risposta. Configurazione consigliata: **medium per l'uso quotidiano, xhigh solo per task complessi**. Su una macchina a 15–25 tok/s, la differenza tra 3K e 10K token di reasoning è la differenza tra risposta in 2 minuti e 7 minuti.

**Conclusione sezione 2:** Qwen3.8-27B è il modello giusto per questo progetto, con due gestione-obblighi (effort di reasoning e assenza di MMLU-Pro) e nessun showstopper. Il valore +14 AA Index è reale e rilevante per il caso d'uso.

---

## 3. Hardware minimo: banda necessaria e classi di macchina

### 3.1 La barra matematica: banda di memoria, non potenza CPU

Il decode autoregressivo di un modello denso è un flusso sequenziale di lettura dei pesi: ogni token generato richiede di leggere **tutti i parametri attivi** dalla memoria. Per Qwen3.8-27B Q4_K_M (15,93 GiB = **17,11 GB decimali** [S][22]):

```text
tok/s max teorici ≈ banda effettiva (GB/s) / 17,1 GB
8 tok/s → banda effettiva ≥ 137 GB/s
12 tok/s → banda effettiva ≥ 205 GB/s
```

[Δ/C] — calcolo aritmetico da specifiche. Con l'efficienza reale dei flussi sequenziali (60–80% del picco, fattore prudente adottato nei report precedenti; l'ancora Strix Halo sotto misura ~77%):

```text
8 tok/s → picco dichiarato ≥ 171–228 GB/s
```

Il margine operativo (KV cache, buffer, copie runtime, OS) suggerisce di fissare la soglia di acquisto a **≥ 200 GB/s di picco dichiarato**, con 256 GB/s come valore comodo. Questo è il criterio che separa le classi: non i core, non il nome dell'iGPU.

Doppio controllo con l'ancora pubblica più vicina [M]: Framework Desktop 395/128 GB (256 GB/s) ha misurato **4,97 tok/s su Llama 3.1 70B** IQ4 (~39,7 GB di pesi) — cioè **~77% di efficienza** (256/39,7 = 6,45 tok/s teorici; misurati 4,97). Qwen3.8-27B Q4 ha pesi ~2,3× più piccoli: la stessa macchina a pari efficienza darebbe **11,5 tok/s**, intervallo di progetto **9–13 tok/s [E]**, sopra soglia ma da misurare. Questa coerenza è il motivo per cui 256 GB/s è la classe credibile.

### 3.2 Memoria necessaria per contesto (pesi + KV + margine)

Da KV = 65.536 byte/token [S][22] e Q4_K_M 15,93 GiB + 0,87 vision [S][22]:

| Contesto | KV cache [Δ/C] | Pesi+KV (GiB) | Minimo macchina confortevole | Note |
|---|---:|---:|---|---|
| 4K | ~0,25 GiB | ~17 | 24 GB (stretto) | fattibile solo con UMA ben gestita |
| 8K | ~0,5 GiB | ~17,3 | 24–32 GB | configurazione tipica agent |
| 32K | ~2 GiB | ~18,8 | **32 GB min, 64 GB comodi** | sessioni lunghe residenti |
| 128K | ~8 GiB | ~25 | **64 GB** | assistente con base documentale |
| 262K (nativo pieno) | ~16 GiB | ~33 | **64 GB min, 128 GB consigliati** | la classe mini lo supera solo con UMA 128 GB |
| 1M (YaRN f4) | ~61 GiB | ~78 | **128 GB (al limite), meglio ≥192** | fuori dalla classe mini; vedere sezione 7 |

Nota YaRN già formalizzata nel report Ranking [22][E]: lo static YaRN factor 4.0 penalizza i prompt corti; per uso tipico 4–16K conviene factor ridotto (es. 2.0 per ~500K) o YaRN spento.

### 3.3 Le quattro classi hardware valutate

Valutazione per classe con il criterio combinato: banda × capacità × quantizzazione Q4_K_M (riferimento) + prefill (TTFT).

**Classe 1 — CPU-only / iGPU su DDR5 dual-channel (780M/890M, 89,6–120 GB/s): RESPINTA per la soglia chat.**

| Piattaforma | Banda teorica [S/C] | tok/s stimati Q4 27B [E] | Verdetto soglia |
|---|---:|---:|---|
| DDR5-5600 dual (7840HS/8845HS/9955HX, 96 GB) | 89,6 GB/s | 3–5 | no |
| DDR5-6400 dual (285HX) | 102,4 GB/s | 3,5–5 | no |
| LPDDR5X-7500 (EVO-X1, 32 GB) | 120 GB/s | 4–6 | no (e capacità insuff.) |

Con 89,6 GB/s il tetto teorico è **5,2 tok/s** anche a efficienza perfetta [Δ]: la soglia è **matematicamente irraggiungibile**, non solo improbabile. L'iGPU non cambia l'equazione (la banda è la stessa RAM). Queste macchine restano utili come host eGPU e per modelli ≤14B.

**Classe 2 — Mini PC + eGPU OCuLink (host 96 GB + GPU discreta): POSSIBILE ma condizionata.**

Con RTX 3090 usata (24 GB, 936 GB/s [S]): i 15,93 GiB dei pesi stanno **interamente in VRAM** con ~8 GB liberi per KV → il decode è limitato dalla banda VRAM (936/17,1 ≈ 55 tok/s teorici [Δ]; ~77% efficienza → **25–45 tok/s [E]**); il prefill di prompt lunghi passa dal link PCIe 4.0×4 (~7,9 GB/s). Rispetto al solo iGPU il salto è enorme, ma: consumo ~350–450 W sistema, rumore, e il costo totale (host+dock+GPU) converge verso 1.100–1.700 €. È la classe giusta se si vuole restare x86/riparabile e si trova una 3090 a buon prezzo. Con una 4060 Ti 16 GB (288 GB/s) i pesi entrano a filo (15,93 < 16) e la KV va in RAM/system: tetto teorico ~17 tok/s [Δ], stima **10–16 tok/s [E]** ma configurazione fragile.

**Classe 3 — UMA ad alta banda (Strix Halo 395/128 GB, 256 GB/s): LA CLASSE MINI PC RISPETTO ALLA SOGLIA.**

- 15,93 GiB di pesi + KV a qualsiasi contesto ≤262K: il fit è comodo su 128 GB [S/Δ].
- Decode stimato **9–13 tok/s [E]** (efficienza ~77% misurata su 70B, sezione 3.1); 8 tok/s è il risultato prudente di test, non garantito.
- Consumo tipico 80–140 W (MS-S1 dichiara 130 W sostenuti [S]); silenzioso; RAM saldata (→ comprare 128 GB, non 64).
- Prezzo: il vero difetto. EU 2.400–4.000 € per i brand noti; CN importato ~2.300–2.700 € (sezione 4).

**Classe 4 — Apple Silicon (M4 Pro 273 GB/s; M2/M4 Max 400–546 GB/s): IL COMPARATORE ANCHE SE NON È UN MINI PC x86.**

- M4 Pro 48/64 GB: banda 273 GB/s [S] → tetto teorico ~16 tok/s [Δ], stima 12–15 tok/s [E]; 64 GB comodo per 262K; silenzioso, ~40–70 W.
- Mac Studio M2 Max usato (400 GB/s, 64–96 GB): tetto teorico ~23 tok/s [Δ], stima 14–20 tok/s [E]; ~35–100 W; prezzo usato 1.200–1.600 € [€].
- Ecosistema chiuso, RAM saldata, e il runtime è Metal/MLX/llama.cpp-Metal: la compatibilità del GGUF specifico di Qwen3.8 (architettura ibrida Gated DeltaNet, molto recente) **va verificata esplicitamente prima dell'acquisto** [E] — è il rischio principale di questa classe, prima ancora del prezzo.

### 3.4 Tabella di sintesi per classe

| Classe | Banda picco | RAM/VRAM utile | Q4 27B: tok/s [E] | TTFT caldo 512 tok [E] | Soglia ≥8? | Costo completo [€] |
|---|---:|---:|---:|---:|---|---:|
| CPU-only dual-channel 96 GB | 89,6–102,4 GB/s | 96 GB | 3–5 | 1–3 s | **no (matematico)** | 600–1.100 |
| Host + eGPU 3090 | 936 GB/s VRAM + link | 24 VRAM + 96 RAM | 25–45 (pesi in VRAM) | <1–2 s | sì, se GPU buona | 1.100–1.700 |
| Strix Halo 395/128 | 256 GB/s | 128 GB UMA | 9–13 | ≤1–2 s | **sì da certificare** | 2.300–4.000 |
| Apple M4 Pro 64 / M2 Max usato | 273–400 GB/s | 64–96 GB | 12–20 | ≤1 s | sì da certificare | 1.300–1.900 / 1.200–1.600 |
| (Riferimento: dual-P40) | 692 GB/s split 2×24 | 48 GB | 3–6 [E] (70B) | non misurato | no (già respinto) | 850–1.250 + 78–126 €/mese |

Il paradosso visibile nella tabella: la dual-P40 ha *più banda nominale* di ogni mini PC, ma la divide in due dispositivi senza pool unificato, con 16 GB per scheda che non contengono nemmeno il 27B Q4 intero (15,93 GiB + runtime: al limite) e con l'intera macchina che paga 550–750 W per i difetti di topologia. La banda utile non è la banda del datasheet.

**Regola di dimensionamento (riassunto sezione 3):** per Qwen3.8-27B a soglia chat servono **≥ 200 GB/s di banda dichiarata con il modello interamente in un pool di memoria unico, ≥ 32 GB di capacità (64 consigliata), quantizzazione Q4_K_M o superiore**. Tutto il resto è compromesso da validare o sperimentale.

---

## 4. Varianti component-driven e censimento prodotti (marche note e no, EU vs CN)

### 4.1 Metodo

Il filtro è la **combinazione di componenti** (banda/canali memoria + iGPU/dGPU + quantizzazione), non il marchio: le marche sconosciute (OEM/ODM su Taobao/1688/AliExpress) fanno parte del censimento e spesso stanno ai prezzi migliori. La provenienza dei listing è il censimento già verificato in `VARIANTI-70B.md` §3–4 (snapshot settembre 2026); qui è **riletto e rioridinato per il target 27B**, che cambia alcune conclusioni (soprattutto la classe eGPU, perché 15,93 GiB di pesi entrano in 16–24 GB di VRAM, cosa impossibile per il 70B). Il cambio cambia anche la soglia: alcune combinazioni **sotto soglia per il 70B diventano credibili per il 27B**. Le conversioni CN→EU usano il riferimento ECB già adottato (€1 = ¥7,8159; €1 = $1,1652) [C], +10–20% spedizione/intermediario, +22% IVA; dazio da verificare sul codice doganale [C/E].

### 4.2 Combinazioni C1–C7

| Combinazione | Componenti chiave | Banda utile | Stima Q4 27B [E] | Soglia ≥8? | Giudizio |
|---|---|---:|---:|---|---|
| **C1** Strix Halo UMA 128 GB | AI Max+ 395, LPDDR5X-8000 256-bit | 256 GB/s | 9–13 tok/s | sì, da certificare | **candidata principale mini PC** |
| **C2** Apple Silicon 64–96 GB | M4 Pro 273 / M2 Max 400 GB/s | 273–400 GB/s | 12–20 tok/s | sì, da certificare | comparator; rischio compatibilità runtime per arch. ibrida |
| **C3** CPU-only dual-channel 96 GB | 7840HS–8945HS/9955HX + DDR5-5600 | 89,6 GB/s | 3–5 tok/s | **no (matematico)** | solo sperimentale/async |
| **C4** Mini PC + eGPU OCuLink + GPU 16 GB | host 8845HS/HX370 + 4060 Ti 16 GB | ~288 GB/s VRAM | 10–16 tok/s con caveat | sì, fragile | pesi a filo (15,93 < 16), KV su RAM: da testare |
| **C5** Mini PC/host + eGPU + RTX 3090 24 GB | host + 3090 usata | 936 GB/s VRAM | 25–45 tok/s | sì | la più sicura x86; consumo 350–450 W |
| **C6** Workstation/mini-server usato multicanale | Xeon Scalable/EPYC 6–8 canali DDR4 | 128–205 GB/s | 5–10 tok/s | borderline | rumore/consumi; solo se già disponibile a poco |
| **C7** Box con GPU discreta integrata (PCIe interno) | MS-01/MS-A2/MS-02 + GPU | VRAM + slot x8/x16 | come C4/C5 | sì | più costoso ma riparabile |

### 4.3 Censimento prodotti per combinazione

**C1 — Strix Halo 395/128 GB** (prezzi snapshot [€], da `VARIANTI-70B.md` §3.1; identità componenti verificata lì):

| Prodotto | Prezzo EU/occidentale | Prezzo CN osservato | CN importato stimato [Δ/C] | Note |
|---|---:|---:|---:|---|
| Beelink GTR9 Pro | ~$1.809 | **¥12.999** | **~€2.335** | miglior listing CN verificato tra i brand |
| FEVM FAEX1 (1 L, OCuLink) | ~$3.317 (AliExpress varianti) | **¥13.999** | ~€2.513 | miglior rapporto volume/OCuLink; garanzia da verificare |
| GMKtec EVO-X2 128 GB/2 TB | ~€3.230 (EU warehouse) | ¥14.499 | ~€2.600 | già censito EU; import CN marginale |
| MOREFINE H1 (395 PRO) | $2.199–3.199 | **¥14.798** | ~€2.655 | variante PRO da verificare |
| Bosgame M5 | $2.399–2.999 | n.v. | — | test solo con reso |
| Framework Desktop 128 GB | $2.851–3.149 | n.v. | — | ecosistema/riparabilità migliori; prezzo alto |
| Minisforum MS-S1 MAX | ~€3.999 EU | n.v. | — | rete 10GbE/USB4; prezzo workstation |
| PELADN YO2 | ~$3.312–3.868 | ¥20.499–21.999 | ~€3.590 | il vantaggio CN sparisce dopo import |
| AOOSTAR NEX395 | n.v. | ¥19.999 (128 GB) / ¥11.999 (64 GB) | — | 64 GB = fit non robusto; scartare per 27B con 262K |
| X+ XRIVAL (395/**96 GB** LPDDR5X-8533) | ~$1.479 (listing) | n.v. | — | **la variante economica interessante**: 96 GB ~273 GB/s; fit 27B Q4 ok a contesti ≤128K; venditore/resi decisivi |
| Sixunited/BOESIIPC/CoreNest/NextNuc/Topton (OEM) | €1.581–3.100 (listing vari) | Alibaba $1.964–3.112 (MOQ) | caso per caso | lead, non acquisti certi: richiedere part number, foto scheda, BIOS, reso |

Lettura: per il 27B il vincolo "128 GB obbligatori" si rilassa (96 GB bastano fino a ~128K di contesto [Δ]) e questo **apre la fascia XRIVAL/OEM da ~$1.5–1.9k** che per il 70B era esclusa. Resta il vincolo banda: 96 GB su bus 256-bit va verificato realmente (LPDDR5X-8533 dichiarata → 273 GB/s teorici [S/P]).

**C2 — Apple Silicon** (prezzi mercato usato/ricondizionato [€]):

| Prodotto | Banda [S] | Memoria | Prezzo osservato | Nota specifica per Qwen3.8 |
|---|---:|---:|---:|---|
| Mac mini M4 Pro 64 GB | 273 GB/s | 64 GB | ~€1.500–1.900 (variante 64 GB) | fit Q4 + KV fino a 262K stretto ma ok; verificare llama.cpp/Metal sul GGUF ibrido |
| Mac Studio M2 Max usato 64/96 GB | 400 GB/s | 64–96 GB | ~€1.200–1.600 | margine e banda migliori; stesso caveat runtime |
| Mac Studio M4 Max 128 GB | 546 GB/s | 128 GB | ~€3.500+ | fuori dalla logica "meno costoso" |

**C3/C4/C5 — host 8845HS/HX370 + SO-DIMM 96 GB + eGPU** (prezzi da `VARIANTI-70B.md` §3.2 e `ANALISI-MINIPC-COMPONENTI.md` §7): 

| Prodotto host | CPU | OCuLink | Prezzo EU [€] | Prezzo CN [€] | Config. completa stimata [Δ] |
|---|---|---|---:|---:|---:|
| Minisforum UM780 XTX (refurb) | 7840HS | sì | ~€349 box | — | €570–700 senza GPU |
| GMKtec K8 Plus | 8845HS | sì | €500–700 completo | ~¥2.999 barebone (~€383 importato) | €600–800 senza GPU |
| AOOSTAR GEM12 Pro | 8845HS | sì | €600–850 completo | — | €600–850 senza GPU |
| Maxtang H255 | 8745HS | sì | — | ~$297 AliExpress | lead OEM; BIOS/2×48 da verificare |
| Minisforum AI X1 Pro-370 | HX370 | sì | €729 barebone / 96 GB ~€1.800–2.000 | ~¥4.212+ | €1.100–1.900 senza GPU |
| Minisforum MS-01 | i9-13900H | slot PCIe x8 interno | ~€709 box | — | €800–1.000 senza GPU |
| Minisforum MS-A2 | 9955HX | PCIe x8 | ~€839 base | — | €1.100–1.500 senza GPU |

GPU per C4/C5 (prezzo usato/nuovo osservato [€]): RTX 4060 Ti 16 GB €300–550; RTX 3090 usata €550–800 (occasione ~€400–550 locale con test); dock OCuLink €100–250. **Con il 27B la 4060 Ti 16 GB diventa rilevante** (pesi 15,93 GiB < 16 GB [S/Δ]): differenza rispetto al 70B, dove era solo un esperimento. Caveat tecnico onesto: a 16 GB restano <1 GB per KV e buffer compute → serve **KV su RAM/quantizzata o quant Q4_K_S/IQ4_XS** (file ~14–15 GiB [P]) e il comportamento reale va misurato; fascia 10–16 tok/s [E] da validare. Con 3090 24 GB il margine è netto (KV 32K = 2 GiB in VRAM [Δ]) e la stima sale a 25–45 tok/s [E].

**C6 — workstation/server usato multicanale** (menzionato dal task come "DDR5 dual/quad"; nota: il quad-channel client non esiste, esiste su HEDT/server): Xeon Scalable 6 canali DDR4-2666 ≈ 128 GB/s [S/C] → 7–8 tok/s [E] borderline; EPYC 7002/7003 8 canali DDR4-3200 ≈ 205 GB/s [S/C] → 10–13 tok/s [E] ma 200–280 W e rumore da server; solo sensato se un unità usata costa <€800. Non è la raccomandazione, è il confine del filtro.

### 4.4 Come leggere i prezzi CN

Regole già formalizzate in `VARIANTI-70B.md` §4.2 e qui riassunte: un prezzo Taobao/1688 è un annuncio, non una fattura; occorre chiedere per iscritto part number, revisione BIOS, lingua/firmware, alimentatore incluso, video di avvio, policy reso; aggiungere 10–20% logistica + 22% IVA + eventuale dazio; 1688 richiede quotazione live dell'agente (le pagine non sono leggibili stabilmente in modo pubblico — dato già dichiarato lì). Dove il vantaggio CN dopo importazione è <10%, preferire il canale EU con garanzia.

---

## 5. TCO: dual-P40 (baseline) vs mini PC candidati vs cloud

### 5.1 Ipotesi (identiche ai report precedenti, per confrontabilità)

```text
16 h/giorno × 30 giorni = 480 h/mese = 5.760 h/anno
elettricità: 0,25–0,35 €/kWh, caso base 0,30 €/kWh
manutenzione: 5% del capex per anno (ventole, SSD, PSU, resi)
TCO(y) = capex + potenza_kW × 5.760 × 0,30 × y + 5% × capex × y
```

Le potenze sono **medie alla presa** durante l'uso [E], non TDP; nessun valore residuo dell'hardware; calore e climatizzazione NON monetizzati (per la P40 sono un'esternalità reale che peggiora il suo TCO economico, non il contrario).

### 5.2 Baseline dual-P40

| Voce | Valore | Evidenza |
|---|---:|---|
| Capex macchina completa | 850–1.250 €, caso base 1.000 € | [€/E] da REPORT-VERIFICA-INDIPENDENTE |
| Potenza media alla presa | 650 W (intervallo 550–750) | [E] da misurare; 500 W sono già il TDP delle sole GPU [S] |
| Energia annua | 0,65 kW × 5.760 h = 3.744 kWh → **1.123 €/anno** a 0,30 (936 a 0,25; 1.310 a 0,35) | [Δ/C] |
| Manutenzione | 50 €/anno | [E] |
| **Costo ricorrente** | **≈ 1.173 €/anno** | [Δ] |
| Rumore/calore | 650–750 W dissipati in casa, ~2.200–2.560 BTU/h | [E] |
| Prestazione sul 70B | 3–6 tok/s [E]; su Qwen3.8-27B Q4 (se mai caricato: 15,93 GiB su 24 GB/scheda, split layer) forse 4–8 tok/s [E] — ma sarebbe un uso sprecone della macchina | [E] |

TCO P40: **1 anno 2.173 € · 2 anni 3.346 € · 3 anni 4.519 € · 5 anni 6.866 €** [Δ] (coerente con ANALISI-RIGOROSA §5.3).

### 5.3 TCO dei candidati mini PC (capex caso base + 5.760 h/anno)

| Candidato | Capex [€] | Potenza media [E] | Energia/anno [Δ] | Manut./anno | **1 anno** | **2 anni** | **3 anni** | **5 anni** | Soglia chat 27B? |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **Mac Studio M2 Max usato 64/96 GB** | 1.500 (1.200–1.600) | 80 W | 138 € | 75 € | **1.713** | 1.926 | **2.140** | 2.566 | da certificare |
| **Mac mini M4 Pro 64 GB** | 1.700 (1.500–1.900) | 60 W | 104 € | 85 € | **1.889** | 2.077 | **2.266** | 2.643 | da certificare |
| **Host eGPU + RTX 3090 24 GB** (C5) | 1.400 (1.100–1.700) | 420 W | 726 € | 70 € | **2.196** | 2.992 | **3.788** | 5.379 | plausibile [E] |
| **Strix Halo 395/128 GB import CN** (C1) | 2.600 (2.300–4.000) | 140 W | 242 € | 130 € | **2.972** | 3.344 | **3.716** | 4.460 | da certificare |
| Strix Halo 395/128 GB canale EU | 3.500 (3.230–4.000) | 140 W | 242 € | 175 € | 3.917 | 4.334 | 4.751 | 5.585 | da certificare |
| Host + 4060 Ti 16 GB (C4) | 1.200 (1.000–1.500) | 260 W | 449 € | 60 € | 1.709 | 2.219 | 2.728 | 3.746 | fragile, da testare |
| CPU-only UM780/K8 + 96 GB (C3) | 650 (600–850) | 50 W | 86 € | 33 € | 769 | 888 | 1.007 | 1.244 | **no (matematico)** |
| *Dual-P40 (baseline)* | *1.000* | *650 W* | *1.123 €* | *50 €* | *2.173* | *3.346* | *4.519* | *6.866* | *no* |

Ogni 10 W medi in più = 17,3 €/anno a 0,30 €/kWh [Δ] (51,8 €/triennio).

### 5.4 A quale anno il mini PC ripaga la differenza d'acquisto vs P40

Confronto al punto decisionale (**la P40 non è ancora stata acquistata**: è il caso rilevante, dato che il task chiede se rende la stack P40 inutile):

| Candidato | Differenza capex vs P40 (1.000 €) | Risparmio ricorrente/anno vs P40 | **Pareggio** [Δ] |
|---|---:|---:|---|
| Mac Studio M2 Max usato | +500 € | 960 € | **~6 mesi** |
| Mac mini M4 Pro 64 GB | +700 € | 984 € | **~8,5 mesi** |
| Host + 3090 eGPU | +400 € | 377 € | **~13 mesi** |
| Strix Halo 128 GB import CN | +1.600 € | 801 € | **~24 mesi** |
| Strix Halo 128 GB canale EU | +2.500 € | 756 € | **~3,3 anni (~40 mesi)** |
| CPU-only 96 GB | −350 € (costa meno) | 1.054 € | immediato (ma non supera la soglia) |

Sensibilità tariffa (Mac mini M4 Pro): pareggio a 0,25 €/kWh ≈ 10,3 mesi; a 0,30 €/kWh ≈ 8,5 mesi; a 0,35 €/kWh ≈ 7,3 mesi [Δ]. La variazione della tariffa muove il pareggio di ±1,5 mesi: **il risultato è robusto**.

Scenario alternativo, **se la P40 fosse già in funzione** (capex affondato): sostituirla si ripaga con il solo risparmio ricorrente → Mac mini ≈ 21 mesi; Strix Halo ≈ 3,2 anni; host+3090 ≈ 3,7 anni [Δ]. In questo caso la sostituzione va motivata anche con silenzio/calore/spazio, non solo con la bolletta.

### 5.5 Confronto cloud sullo stesso modello

Formula già adottata nei report precedenti: `TCO cloud annuo = h/mese × 12 × €/h × 1,10 + 120 €/anno storage`. Tariffe snapshot [€] da Beam/Lambda/RunPod già censite (A6000 ≈ 0,50 €/h; L40S ≈ 0,70 €/h; A100 80 GB ≈ 1,25 €/h; H100 ≈ 1,68 €/h); su un cloud GPU moderna Qwen3.8-27B Q4 girerebbe con vLLM/SGLang a 30–60+ tok/s e batch [E] — il confronto "costo per risposta utile" favorisce ancora di più il cloud ad alta concorrenza.

| Profilo | GPU | 80 h/mese | 160 h/mese | 480 h/mese (16h/giorno sempre) |
|---|---|---:|---:|---:|
| On-demand/interruptible | A6000 48 GB | 648 €/anno | 1.176 €/anno | 3.288 €/anno |
| On-demand/interruptible | L40S 48 GB | 859 €/anno | 1.598 €/anno | 4.555 €/anno |
| On-demand/interruptible | A100 80 GB | 1.440 €/anno | 2.760 €/anno | 8.040 €/anno |
| On-demand/interruptible | H100 80 GB | 1.894 €/anno | 3.668 €/anno | 10.765 €/anno |
| API a token (provider serverless dello stesso modello) | — | dipende dal volume: `input×tariffa_in + output×tariffa_out`; tariffe specifiche per Qwen3.8-27B **non verificate qui** → non si usano numeri | idem | idem |

Lettura economica (caso base Mac mini M4 Pro: 1.889 € il primo anno, 188 €/anno ricorrenti [Δ]):

- **Uso ≤ ~100 h/mese**: il cloud on-demand costa meno di qualunque mini PC anche dopo anni (A6000 a 80 h/mese = 648 €/anno vs 1.889 € primo anno del mini). Vince il cloud, a parità di modello.
- **Uso ~160 h/mese**: cloud A6000 1.176 €/anno vs mini 1.888+188×y → pareggio tra il 1º e il 2º anno; da lì il locale vince [Δ].
- **Uso 480 h/mese (il profilo dichiarato, 16h/giorno)**: cloud 3.288–10.765 €/anno **per sempre**, mini PC 188–372 €/anno ricorrenti per i candidati UMA (l'host+3090 sale a ~796 €) → tutti i candidati raccomandati vincono già nel primo anno contro la tariffa cloud più economica censita (A6000 a 3.288 €); eccezione lo Strix canale EU a prezzo pieno (3.917 € al 1º anno). Il cloud always-on a 480 h/mese è economicamente irrazionale per un utente singolo (stessa conclusione dei report precedenti, ora rafforzata: con il 27B la macchina locale che supera la soglia **costa 5–10× meno** del P40-project).

Nota anti-truffa del confronto: il cloud a 0,50 €/h offre ~30–60 tok/s [E] con serving moderno, il mini PC 10–20 [E]: a parità di token prodotti il cloud a 480 h/mese resta 3–8× più caro, ma a uso sporadico il costo per token può favorire il cloud. **La decisione dipende dalle ore reali, non dal prezzo orario.**

### 5.6 Sintesi TCO

1. La dual-P40 è dominata su ogni asse: capex simile o superiore, ricorrenze 3–6× superiori, soglia non superata, rumore/calore. In TCO: i Mac battono la P40 **già al 1º anno** (1.713/1.889 vs 2.173 €), i path economici (host+GPU, CPU-only) entro il 1º–2º, lo Strix importato CN pareggia al 2º anno (3.344 vs 3.346 €); **solo lo Strix canale EU a prezzo pieno non batte la P40 prima del 5º anno** — ulteriore argomento contro il canale EU senza sconto.
2. Il vero trade-off residuo è **tra i candidati sopra soglia**: capex (Strix 2.300–4.000) vs rischio (4060 Ti fragile) vs ecosistema (Mac chiuso ma economico da far girare).
3. Il cloud vince solo sotto ~100–160 h/mese di uso reale; con 16 h/giorno dichiarate, il locale è la scelta razionale — purché il test di accettazione (sezione 8) certifichi la soglia.

---

## 6. Matrice decisionale pesata e raccomandazione

### 6.1 Pesi (criteri dell'utente: costo, consumo, silenzio, qualità sufficiente, semplicità)

| Criterio | Peso | Justificazione |
|---|---:|---|
| Chat: decode ≥8 tok/s + TTFT | 30% | soglia di progetto, non negoziabile per l'uso primario |
| TCO a 3 anni (16 h/giorno) | 15% | il punto del task: meno costoso del P40 |
| Consumo elettrico | 15% | bolletta + calore domestico |
| Silenzio/calore | 15% | macchina in casa, 16h/giorno |
| Qualità sufficiente del modello (profilo agentic) | 10% | Qwen3.8 vs alternative 27–32B per il caso d'uso |
| Semplicità/manutenzione | 10% | server sempre acceso, nessun airflow custom |
| Upgrade/future-proofing | 5% | RAM sostituibile, GPU, runtime |
| **Totale** | **100%** | |

Punteggi 1–5, con la regola anti-overclaim già usata nei report precedenti: un punteggio alto senza benchmark sul target esatto è vietato; l'incertezza penalizza. Il modello è fisso (Qwen3.8-27B Q4), quindi la riga "qualità" discrimina soprattutto la capacità di gestire effort xhigh e contesto.

### 6.2 Matrice

| Percorso | Chat 30 | TCO 15 | Energia 15 | Silenzio 15 | Qualità 10 | Semplicità 10 | Upgrade 5 | **Totale /100** | Esito |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Mac Studio M2 Max usato 64/96 | 3 | 4 | 5 | 5 | 4 | 5 | 2 | **80** | **prima scelta se reperibile bene usato** |
| **Mac mini M4 Pro 64 GB** | 3 | 4 | 5 | 5 | 4 | 5 | 1 | **79** | **prima scelta assoluta, salvo vincolo x86** |
| **Strix Halo 395/128 GB (import CN)** | 3 | 3 | 4 | 5 | 5 | 4 | 2 | **74** | **raccomandato x86, con reso** |
| Strix Halo 395/128 GB (canale EU) | 3 | 2 | 4 | 5 | 5 | 4 | 2 | **71** | alternativa se CN sconsigliato; TCO penalizzante |
| CPU-only dual-channel 96 GB (C3) | 1 | 5 | 5 | 5 | 3 | 5 | 4 | **71** | **escluso dal filtro duro chat**; solo async/sperimentale |
| Host + RTX 3090 eGPU (C5) | 4 | 2 | 1 | 1 | 5 | 2 | 5 | **55** | se serve x86 + margine prestazionale |
| Host + 4060 Ti 16 GB (C4) | 2 | 3 | 2 | 2 | 4 | 2 | 4 | **49** | esperimento da validare |
| Workstation EPYC/Xeon multicanale (C6) | 2 | 2 | 1 | 1 | 4 | 2 | 4 | **40** | scartata |
| Dual-P40 (baseline 70B) | 1 | 1 | 1 | 1 | 3 | 1 | 1 | **24** | demolita |
| Cloud on-demand A6000/L40S (stesso modello) | 5 | 3 intermittente / 1 a 480h | 5 | 5 | 5 | 5 | 5 | **94 intermittente / 88 sempre-on** | riferimento qualità; vince sotto ~160 h/mese |

Note di lettura: **filtro duro** — un percorso sotto la soglia chat (CPU-only, C6, P40) è escluso dalla raccomandazione primaria indipendentemente dal punteggio totale, perché nessun risparmio compensa un requisito mancante (regola già adottata in ANALISI-RIGOROSA §8). I Mac vincono la matrice complessiva perché il caso d'uso primario (Qwen3.8 Q4, contesti ≤262K) non richiede upgrade hardware, e la penalità "upgrade" pesa solo 5%. Lo Strix Halo resta **la raccomandazione se il vincolo è x86/Linux/riparabilità o l'uso oltre il 27B** (modelli >40B futuri). I punteggi chat = 3 (non 4–5) riflettono l'assenza di misura pubblica del Qwen3.8-27B su queste piattaforme: la promozione a 4–5 avviene solo dopo il test di sezione 8. Il punteggio cloud dipende interamente dalle ore: 94 intermittente, 88 a 480 h/mese (dove il TCO crolla a 1).

### 6.3 Raccomandazione quantitativa

1. **Prima scelta (nessun vincolo x86): Mac Studio M2 Max usato 64/96 GB con reso (~1.200–1.600 € [€]) oppure Mac mini M4 Pro 64 GB nuovo (~1.500–1.900 € [€]).** Capex ≤ 1.700 €, ricorrenze ~190–215 €/anno, silenzioso, banda 273–400 GB/s, supera la stima di banda con margine. Rischio principale da testare prima: compatibilità del GGUF ibrido Qwen3.8 su llama.cpp/Metal/MLX (sezione 8).
2. **Prima scelta x86/riparabile: Strix Halo 395/128 GB importato CN con reso (~2.300–2.700 € [€]), candidati in ordine: Beelink GTR9 Pro, FEVM FAEX1, GMKtec EVO-X2, MOREFINE H1.** Se il canale CN non dà reso → canale EU (~3.230–4.000 €) o il confronto con il Mac diventa squalificante.
3. **Variante economica x86 con margine prestazionale: host 96 GB + RTX 3090 usata via OCuLink/PCIe (~1.400 € [€])** — supera strettamente la soglia stimata ma consumo/rumore la declassano rispetto al profilo dichiarato (silenzio, 16h/giorno).
4. **NON comprare:** dual-P40 (già respinta); mini PC dual-channel 96 GB come macchina chat 27B (limite matematico di banda); eGPU da 16 GB come deployment (la 4060 Ti 16 GB va considerata solo come test a basso costo); Strix Halo 64 GB (RAM saldata insufficiente per il profilo contesti lunghi); workstation multicanale rumorose.
5. **Il modello è adottato:** Qwen3.8-27B come primario locale (chat medium, agentic xhigh), Qwen3-32B o Gemma 3 27B come fallback per conoscenza pura finché manca MMLU-Pro omogeneo; cloud on-demand per i picchi di qualità al di sopra della classe locale.

---

## 7. Quando la scelta cambia (scenari)

| Scenario | Cosa cambia | Nuova risposta |
|---|---|---|
| **Serve 262K pieno con sessioni multiple residenti** | KV 16 GiB per sessione [S/E][22]; 1M ≈ 61 GiB | Serve UMA/GPU ≥64 GB con margine → Strix Halo 128 GB resta ok per 1–2 sessioni 262K; Mac 64 GB stretto; 1M full: solo Apple 128+ GB o cloud A100/H100. Se il 1M è requisito → il mini PC non è più il problema, il cloud torna competitivo |
| **Il contesto tipico è 4–8K (agent con tool, non documenti lunghi)** | KV trascurabile (≤0,5 GiB [Δ]) | La barra si abbassa: entra anche la **variante 96 GB Strix (XRIVAL ~$1.479 [€])**, la 4060 Ti 16 GB diventa deployment credibile dopo test, e il Mac mini M4 Pro 48 GB (capacità stretta, banda ok). Il "mini economicissimo" (host + 4060 Ti, ~1.000–1.300 €) diventa la risposta se il test passa |
| **Solo chat breve, quality threshold rilassata a 4–5 tok/s accettabile** | soglia ribassata | CPU-only dual-channel 96 GB (600–850 €) diventa sufficiente [E] — ma allora si perde la qualità agentic xhigh per latenza; consigliato medium/non-reasoning |
| **Il GGUF ibrido Qwen3.8 non gira bene su Metal/MLX (bug/assenza kernel)** | rischio Mac concretizzato | Passa tutto su Strix Halo (llama.cpp Vulkan/ROCm su 8060S) o C5 eGPU. **Verifica da fare prima dell'acquisto Mac** |
| **La misurazione Strix Halo conferma <8 tok/s a sforzo (ipotesi non remota: il 70B misurava 4,97)** | soglia fallita sulla classe 256 GB/s | Unico percorso locale sopra soglia: C5 (3090 in eGPU/PCIe). Altrimenti cloud on-demand + mini PC async |
| **L'uso reale scende sotto 100 h/mese** | il TCO si ribalta | Cloud on-demand (A6000/L40S) dello stesso modello + mini PC economico per embedding/async. Non comprare il box da 2.5k+ |
| **Serve più di un utente / batch alto** | llama.cpp single-stream non basta | Il cloud con vLLM/SGLang vince per costruzione; locale solo se privacy assoluta |
| **Compare un MoE 27–35B-A3B della stessa generazione con qualità simile (es. Qwen3.6-35B-A3B)** | banda necessaria crolla (~3B attivi) | Si riapre la classe dual-channel/mini economico: file ~20–24 GB, decode 1,5–2× più veloce a pari banda [E][21] — monitorare i rilasci prima dell'acquisto definitivo |
| **Prezzo Strix Halo 128 GB crolla (fase di ciclo ok, sconti CN)** | capex | La raccomandazione x86 si sposta su Strix anche senza vincolo Linux; sotto ~1.800 € importato supera il Mac come scelta default |
| **Il task agentic primario richiede visione (computer-use da screenshot)** | +0,87 GiB pesi, richer stack | Tutto il sizing regge (+0,87 GiB [S][22]); verificare il runtime multimodale locale (llama.cpp mmproj) su ogni piattaforma prima dell'acquisto |

---

## 8. Piano di verifica empirica, fonti e limiti residui

### 8.1 Cosa misurare su un box reale (protocollo fisso, dal test di accettazione dei report precedenti, adattato al 27B)

**Setup fisso (identico su ogni piattaforma, per comparabilità):**

1. File: `Qwen3.8-27B-Q4_K_M.gguf` (15,93 GiB [+0,87 vision]; SHA-256 annotato); secondo file: IQ4_XS/Q4_K_S se la piattaforma è a 16 GB (4060 Ti).
2. Runtime: llama.cpp build recente, backend esplicito (Vulkan/ROCm su Strix; Metal su Apple; CUDA su eGPU); commit annotato; variante `--jinja`/template chat del model card.
3. Contesto: 4.096 baseline, poi 8.192 e 32.192; un solo slot; YaRN spento o factor ridotto (default f4.0 penalizza i prompt corti [22][E]).
4. Reasoning effort: **tre profili** — non-reasoning, medium, xhigh — misurati separatamente (la verbosity xhigh 3,3× [P] altera tutto se non isolata).
5. Tre run dopo warm-up; scartare il primo; riportare media/min/max.

**Metriche di accettazione:**

| Metrica | Soglia promozione |
|---|---|
| Decode tok/s (contesto 4K, medium) | **≥ 8** (nessun calo sotto 6 per >10 s) |
| TTFT caldo, prompt 512 | **≤ 2 s** (P95) |
| TTFT caldo, prompt 4K | ≤ 5 s (P95) |
| Prefill tok/s (prompt 4K) | riportato (non è la soglia, ma determina l'uso agent) |
| Decode durante contesa (chat + job async 2.000 token) | ≥ 8 tok/s mantenuti |
| Memoria libera dopo caricamento | ≥ 8 GB (per KV crecita/seconda sessione) |
| Stabilità | nessun errore driver/Vulkan/ROCm/Metal; 60 min di decode continuo |
| Watt alla presa (idle / prefill / decode) | riportati; verificare coerenza col TCO di sezione 5 |
| Rumore a 50 cm (idle/carico) | riportato |
| Temperatura dopo 60 min | riportata; nessun throttling colante sotto soglia |

**Test specifici per piattaforma:**

- **Strix Halo:** verificare quota UMA assegnabile, frequenza LPDDR5X reale (8000, non fallback), sostenibilità 130 W per 60 min.
- **Mac:** prima dell'acquisto, caricare il GGUF con llama.cpp/Metal (o convertire con MLX se supportato) su un MacBook in store/conosciuto: è l'unico modo di chiudere il rischio compatibilità dell'architettura ibrida Gated DeltaNet.
- **eGPU:** verificare link PCIe 4.0×4 negoziato (non fallback), decodifica con 0/50%/100% layer offload, e consumo del dock.
- **Ogni box importato CN:** memtest notturno, BIOS revision, video di avvio, reso scritto.

**Criterio di rifiuto:** sotto soglia su un solo punto bloccante (decode, TTFT P95, stabilità) → scartare la piattaforma, non abbassare la soglia a posteriori. Il risultato va registrato e conservato come baseline del progetto.

### 8.2 Fonti

**Modello (già censite in Appendice A di RANKING-MODELLI-FINETUNING.md, riportate qui come riferimento primario):**

22. [Qwen/Qwen3.8-27B — model card HF](https://huggingface.co/Qwen/Qwen3.8-27B) e [Qwen3.6-27B README](https://huggingface.co/Qwen/Qwen3.6-27B/raw/main/README.md) — architettura, Q4/FP8/BF16, KV/token, tabelle launch [S]
23. [Kingy.ai — Qwen3.8-27B specs, benchmarks, local hardware](https://kingy.ai/blog/qwen3-8-27b-specs-benchmarks-local-hardware/) — dimensioni file, KV 65.536 B/token, caveat harness [S/E]
24. [Qubrid AI — Qwen3.8-27B benchmarks: official and independent results](https://www.qubrid.com/blog/qwen38-27b-benchmarks-official-and-independent-results) — AA Intelligence Index 52/44/35, Agentic Index ≈51, verbosity xhigh [M/P]

**Hardware (già censite nei report precedenti; qui si cita la fonte primaria):**

25. [AMD Ryzen AI Max+ 395 — specifiche](https://www.amd.com/en/products/processors/laptop/ryzen/ai-300-series/amd-ryzen-ai-max-plus-395.html) — 16C/32T, LPDDR5X-8000 256-bit, 128 GB max [S]
26. [Apple Mac mini M4/M4 Pro](https://www.apple.com/newsroom/2024/10/apples-new-mac-mini-is-more-mighty-more-mini-and-built-for-apple-intelligence/) e [Mac Studio M2 Max specs](https://support.apple.com/en-us/111835) — 273/400 GB/s [S]
27. [Geerling AI/LLM benchmarks](https://github.com/geerlingguy/ai-benchmarks) e [issue Framework Desktop 395/128 GB](https://github.com/geerlingguy/ai-benchmarks/issues/21) — Llama 3.1 70B 4,97 tok/s su Strix Halo [M]
28. [Minisforum MS-S1 MAX](https://store.minisforum.com/products/minisforum-ms-s1-max-mini-pc) — 130 W sostenuti, UMA 128 GB [S]
29. [Beelink GTR9 Pro](https://www.bee-link.com/products/beelink-gtr9-pro-amd-ryzen-ai-max-395) e [FEVM FAEX1 (TechPowerUp)](https://www.techpowerup.com/343971/fevm-squeezes-ryzen-ai-max-395-apu-into-faex1-models-1-liter-enclosure) — listing CN ¥12.999 / ¥13.999 [€]
30. [Liliputing — censimento mini PC AI Max 395 128 GB](https://liliputing.com/more-ryzen-ai-max-395-mini-pcs-with-128gb-are-now-available-if-you-can-afford-one/) — più listing CN/EU [€]
31. [Minisforum UM780 XTX](https://minisforumpc.eu/products/um780-xtx), [GMKtec K8 Plus](https://www.gmktec.com/products/gmktec-nucbox-k8-plus-mini-pc-amd-ryzen-7-8845hs), [AOOSTAR GEM12 Pro](https://aoostar.com/products/aoostar-gem12-amd-r7-pro-8845hs-mini-pc) — host OCuLink economici [S/€]
32. [OCuLink PCIe 4.0×4 (ADT-Link)](https://www.adt.link/product/F9GV4.html) e [OCuLink hands-on](https://www.xda-developers.com/oculink-egpu-hands-on/) — banda link [S/P]
33. [NVIDIA RTX 4060 Ti](https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4060-4060ti/) e [RTX 3090](https://www.nvidia.com/en-us/geforce/graphics-cards/30-series/rtx-3090-3090ti/) — 16 GB/288 GB/s/165 W; 24 GB/936 GB/s [S]
34. [NVIDIA Tesla P40 datasheet](https://images.nvidia.com/content/pdf/tesla/184427-Tesla-P40-Datasheet-NV-Final-Letter-Web.pdf) — baseline 24 GB/346 GB/s/250 W [S]
35. [ECB EUR/CNY](https://www.ecb.europa.eu/stats/policy_and_exchange_rates/euro_reference_exchange_rates/html/eurofxref-graph-cny.en.html) — cambio 9 set 2026 [C]
36. Cloud (tariffe snapshot già censite): [Beam pricing](https://www.beam.cloud/pricing), [Lambda pricing](https://lambda.ai/pricing), [RunPod pricing](https://www.runpod.io/pricing), [Vast.ai](https://vast.ai/pricing) [€]

### 8.3 Limiti residui (dichiarati, non nascosti)

1. **Nessun benchmark riproducibile** esiste oggi per Qwen3.8-27B GGUF su Strix Halo/M4 Pro/mini PC: tutte le cifre di throughput di questo report sono [E] ancorate a proxy (4,97 tok/s su 70B per 395; architettura Apple) e all'aritmetica di banda. La promozione finale richiede il test di §8.1.
2. **MMLU-Pro di Qwen3.8-27B è N.D.** nella forma omogenea; le cifre di terze parti esistono ma non sono state usate. Qualsiasi giudizio di qualità enciclopedica resta incompleto finché Qwen non pubblica la tabella con protocolli allineati.
3. **I benchmark agentic sono vendor-run** con judge GPT-4o/GPT-5.4 e task set corretti: sono [C], non repliche indipendenti. L'unico indizio indipendente è l'AA Index (+14 vs 3.6) [M].
4. **I prezzi sono snapshot** (settembre 2026) di listing e listini; i prezzi CN diventano costi reali solo dopo IVA/dazio/logistica/intermediario, e i listing OEM vanno verificati SKU per SKU. Nessun prezzo qui è un'offerta vincolante.
5. **Le tariffe cloud cambiano** e non includono storage/egress/tasse; l'analisi cloud usa le tariffe snapshot già censite nei report precedenti e la formula condivisa, non un preventivo live.
6. **Il consumo dei candidati è stimato** [E], non misurato alla presa sulle configurazioni finali; il TCO va ricalcolato dopo il test.
7. **Non ho verificato la disponibilità effettiva a stock** dei box CN (listing ≠ stock); per 1688 serve un agente con quotazione live, come già dichiarato in VARIANTI-70B.md.
8. **Compatibilità runtime:** l'architettura ibrida Gated DeltaNet + MTP è molto recente; il supporto llama.cpp/MLX/GGUF può essere parziale o lento su quant specifiche. Va verificato per ogni piattaforma prima dell'acquisto (è il rischio #1 del percorso Mac).

### 8.4 Chiusura

Il cambio di direzione richiesto dal task è **quantitativamente fondato**: passare da "70B su dual-P40 (850–1.250 € + ~1.173 €/anno di elettricità, 3–6 tok/s [E], 550–750 W)" a "Qwen3.8-27B su macchina UMA ≥256 GB/s (1.200–2.700 €, ~190–250 €/anno, 9–13 tok/s stimati [E], silenziosa)" migliora simultaneamente capex/opex/latenza/quality-profile del caso d'uso agentic, con il +14 AA Index del 3.8 sul 3.6 a parità di architettura come giustificazione del modello e l'aritmetica di banda (≥200 GB/s per la soglia) come giustificazione dell'hardware. I due rischi aperti — verifica della soglia reale su Strix/Mac e MMLU-Pro N.D. — sono gestibili con il protocollo di §8.1 e con un modello fallback, e non ribaltano il verdetto.

**Verdetto finale: SÌ alla sostituzione. Acquistare prima un solo box con diritto di reso (Mac Studio M2 Max usato o Strix Halo 395/128 GB), eseguire il protocollo di §8.1, e solo dopo promuovere la piattaforma a stack primario; la dual-P40 e ogni percorso 70B locale vengono archiviati.**
