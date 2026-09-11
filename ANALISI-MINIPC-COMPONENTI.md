# Analisi Mini PC per componenti, modello e costo

**Oggetto:** selezione di Mini PC/workstation compatte per inferenza locale di Qwen2.5-72B e Llama 3.1 70B quantizzati 4-bit.  
**Profilo di progetto:** 16 ore/giorno, chat prioritaria, elaborazioni asincrone secondarie, silenzio e consumi importanti.  
**Criterio di accettazione:** almeno **8 tok/s di decode** a concorrenza 1 e **TTFT caldo ≤2 s** con prompt da 512 token; con 4k token, TTFT target ≤5 s. Un risultato di 3–4 tok/s può essere utile per esperimenti, ma è respinto come soluzione primaria.  
**Data prezzi:** snapshot delle pagine commerciali e dei risultati di ricerca consultati per questa analisi; i prezzi cambiano, possono essere promozionali e non sono un’offerta vincolante.

> **Conclusione anticipata.** La categoria “Mini PC” non è omogenea. Un box Ryzen 7840HS con due SO-DIMM a 5600 MT/s, un Mac Studio M2 Max a 400 GB/s e uno Strix Halo a 256 GB/s non sono la stessa cosa. Tuttavia la memoria veloce non basta da sola: per il target 70B il percorso CPU-only dei comuni Mini PC resta sotto la soglia, mentre lo Strix Halo ha finalmente memoria sufficiente e un iGPU serio ma un benchmark pubblico sul 70B ancora borderline. La shortlist reale è: **MS-S1 MAX/EVO-X2 Strix Halo da testare con reso**, **MS-A2/MS-02 solo se si aggiunge una GPU discreta**, e **UM780/UM890/GEM12/K8 soprattutto come host compatto per eGPU o per modelli inferiori**. Nessun comune Mini PC CPU-only viene promosso senza una misura sul file GGUF esatto.

---

## 1. Metodo, legenda e cosa significa “banda effettiva”

### 1.1 Legenda dell’evidenza

- **[S] Specifica:** dato dichiarato da produttore o documentazione primaria.
- **[C] Calcolo:** prodotto o conversione aritmetica da specifiche dichiarate.
- **[M] Misura:** benchmark pubblicato con hardware e metrica leggibili.
- **[P] Proxy:** misura reale, ma con modello, runtime, quantizzazione o macchina non identici.
- **[E] Stima:** intervallo progettuale; non è una promessa e va verificato.
- **[€] Mercato:** prezzo osservato; non necessariamente prezzo finale italiano.

Le sigle sono applicate anche ai singoli numeri nelle tabelle. Un prezzo [€] non diventa [S] perché appare sul sito ufficiale: resta il prezzo di una variante in quel momento. Un numero “tokens/s” senza distinguere prompt processing da decode non è usato come prova di chat.

### 1.2 Modello e memoria di riferimento

Per Qwen2.5-72B-Instruct i riferimenti pratici sono:

| File 4-bit | Dimensione indicativa | Conseguenza |
|---|---:|---|
| IQ4_XS GGUF | circa 39,7 GB | il minimo ragionevole per 70–72B con margine |
| Q4_0 GGUF | circa 41,4 GB | ancora gestibile con 64/96 GB di RAM, ma la KV cache occupa spazio |
| Q4_K_M GGUF | circa 47,4 GB | stretto su 64 GB; più sicuro su 96/128 GB |
| Llama 3.1 70B Q4_K_M | circa 42,5 GB | riferimento alternativo, stessa classe di memoria |

Le dimensioni sono quelle dei repository GGUF già esaminati nel report precedente e sono approssimazioni: header, runtime, KV cache, contesto e memoria del sistema devono essere aggiunti. “Il file entra nella RAM” non equivale a “il modello è utilizzabile a 8 tok/s”.

### 1.3 Calcolo corretto della banda

Per DDR/LPDDR si usa:

```text
banda teorica = numero di canali × MT/s × 8 byte
```

Esempi:

- DDR5-5600 dual-channel: `2 × 5.600 × 8 = 89,6 GB/s`;
- DDR5-5200 dual-channel: `83,2 GB/s`;
- DDR5-4800 dual-channel: `76,8 GB/s`;
- LPDDR5X-7500 dual-channel: `120 GB/s`;
- LPDDR5X-8000 dual-channel: `128 GB/s`;
- LPDDR5X-8000 su bus 256-bit/quad-channel: `256 GB/s`;
- Xeon E5-2680 v4, DDR4-2400 quad-channel: `4 × 2.400 × 8 = 76,8 GB/s`;
- due Tesla P40: `2 × 346 = 692 GB/s` teorici di VRAM, non disponibili come una singola memoria condivisa e soggetti al traffico PCIe.

Questi sono picchi teorici. Per la pianificazione CPU-only uso, quando serve, un intervallo molto prudente del **60–80%** del picco in un flusso sequenziale; è [E/C], non una misura del Mini PC. Per iGPU la banda è contesa fra CPU, GPU, sistema operativo e copie Vulkan/ROCm: non bisogna moltiplicare automaticamente il picco per il numero di compute unit.

Una 780M o 890M non possiede VRAM dedicata: la memoria “condivisa” è RAM di sistema e la quantità realmente utilizzabile dipende da firmware, driver, OS e runtime. Una configurazione Windows che mostra una piccola quota “dedicated/shared GPU memory” non dimostra che il modello non possa essere allocato in Vulkan; dimostra solo che il budget del driver non è uguale alla RAM fisica. Per un test serio si deve usare Linux/Vulkan o ROCm quando supportato, osservare l’allocazione reale e lasciare margine al sistema.

---

## 2. Componenti e famiglie: la graduatoria che mancava

### 2.1 Tabella comparativa primaria

| Modello e variante | CPU/architettura | RAM e canali realmente documentati | Banda teorica | iGPU | Upgrade/espansione | Prezzo snapshot, box o base | Evidenza |
|---|---|---|---:|---|---|---:|---|
| **Minisforum MS-01 S1390** | Core i9-13900H, 6P+8E/20T, Raptor Lake, AVX2 | 2× SO-DIMM DDR5-5200, ufficiale fino a 96 GB | **83,2 GB/s** | Iris Xe, 96 EU, solo dual-channel | 3× M.2, slot fisico PCIe 4.0 x16 a x8; non OCuLink | circa **€709** EU per il box base | [S][€] |
| **Minisforum UM780 XTX** | Ryzen 7 7840HS, 8C/16T, Zen 4, AVX-512 | 2× SO-DIMM DDR5-5600, 2×48 GB dichiarati/riportati, 96 GB | **89,6 GB/s** | Radeon 780M, 12 CU | OCuLink, 2× M.2 PCIe 4.0 | circa **€349 refurb**, €390–650 nuovo/stock | [S][P][€] |
| **Minisforum UM790 Pro** | Ryzen 9 7940HS, 8C/16T, Zen 4, AVX-512 | ufficiale spesso 64 GB; 2×48 GB funziona in report utenti, non è sempre garantito dalla pagina del modello | **89,6 GB/s** a 5600 | Radeon 780M, 12 CU | 2× M.2, USB4; niente OCuLink nativo | circa **€379 barebone**, €750–930 configurato | [S][P][€] |
| **Minisforum UM890 Pro** | Ryzen 9 8945HS, 8C/16T, Zen 4, AVX-512 | 2× SO-DIMM DDR5-5600, ufficiale fino a 96 GB | **89,6 GB/s** | Radeon 780M, 12 CU | OCuLink, 2× M.2, USB4 | circa **€489** con 32 GB/1 TB; meno per barebone | [S][€] |
| **Minisforum MS-A1 + Ryzen 7 8700G** | 8C/16T Zen 4 desktop, AVX-512, socket AM5 | DIMM DDR5; capacità dipendente da scheda/BIOS, 96 GB plausibile ma da kit-testare | 89,6 GB/s a DDR5-5600 | Radeon 780M | socket AM5, PCIe desktop, slot M.2; percorso eGPU più serio | da circa **$239,90 barebone**, CPU/RAM/SSD extra | [S][€][E] |
| **Minisforum MS-A2 9955HX** | Ryzen 9 9955HX, 16C/32T, Zen 5, AVX-512 | 2× SO-DIMM DDR5-5600, fino a 96 GB | **89,6 GB/s** | Radeon 610M, 2 CU circa | PCIe fisico x16 elettrico x8, split 2×x4; 3× M.2/U.2 | circa **€839** per variante base/box; configurazioni fino a ~€2.100 | [S][€] |
| **Minisforum MS-02 Ultra 285HX** | Core Ultra 9 285HX, 8P+16E/24T, Arrow Lake, AVX2 | 4 slot SO-DIMM, fino a 256 GB ECC; sempre due canali di memoria | 102,4 GB/s a DDR5-6400 | Intel Graphics, 4 Xe-core | PCIe 5.0 x16, 4× M.2, dual 25GbE | variante iniziale da circa **$599** per CPU inferiore; 285HX da quotare | [S][€][E] |
| **Minisforum AI X1 Pro-370** | Ryzen AI 9 HX 370, 4 Zen 5 + 8 Zen 5c/24T, AVX-512 | 2× SO-DIMM DDR5-5600, pagina europea fino a 128 GB; 2×48 da verificare sul kit | **89,6 GB/s** a DDR5-5600 | Radeon 890M, 16 CU | OCuLink, 3× M.2, 2× USB4, 2.5GbE | circa **€729 barebone**, €1.639 per 64 GB/1 TB, 96 GB spesso ~€1.800–2.000 | [S][€][E] |
| **Minisforum MS-S1 MAX** | Ryzen AI Max+ 395, 16C/32T Zen 5, AVX-512 | LPDDR5X-8000 saldata, 128 GB, 256-bit/quad-channel | **256 GB/s** | Radeon 8060S, 40 CU | 128 GB non sostituibili; 2× M.2; slot fisico x16 ma elettrico PCIe 4.0 x4 | circa **$3.799**; EU 64 GB circa €2.679, 128 GB tipicamente **€3.300–4.000** | [S][€] |
| **Beelink SER8 8845HS** | Ryzen 7 8845HS, 8C/16T Zen 4, AVX-512 | 2× SO-DIMM DDR5-5600; sito dichiara fino a 256 GB, 2×48 da verificare | **89,6 GB/s** | Radeon 780M, 12 CU | 2× M.2; USB4 ma niente OCuLink nativo documentato | circa **$799** pagina ufficiale; €550–850 a seconda della variante | [S][€][E] |
| **GMKtec K8 Plus** | Ryzen 7 8845HS, 8C/16T Zen 4, AVX-512 | 2× SO-DIMM DDR5-5600; pagina GMKtec fino a 96 GB | **89,6 GB/s** | Radeon 780M, 12 CU | OCuLink, M.2 PCIe 4.0 | da circa **$399,99**; €500–700 configurato/importato | [S][€] |
| **GMKtec EVO-X1** | Ryzen AI 9 HX 370, 12C/24T, Zen 5/5c, AVX-512 | variante tipica 32 GB LPDDR5X-7500 saldata; non è la stessa macchina dell’AI X1 Pro SO-DIMM | **120 GB/s** teorici | Radeon 890M, 16 CU | RAM saldata, M.2; OCuLink secondo variante | circa **€1.030** EU per 32 GB/1 TB, $1.149,99 listino | [S][€] |
| **GMKtec EVO-X2** | Ryzen AI Max+ 395, 16C/32T Zen 5 | LPDDR5X-8000, 64 o 128 GB saldati, 256-bit | **256 GB/s** | Radeon 8060S, 40 CU | nessun upgrade RAM; storage/USB4, chassis più grande | circa **€3.230** EU per 128 GB/1 TB; pagina USA da $2.199,99 | [S][€] |
| **AOOSTAR GEM12/GEM12 Pro** | Ryzen 7 8845HS/PRO, 8C/16T Zen 4, AVX-512 | 2× SO-DIMM DDR5-5600, GEM12 Pro fino a 128 GB | **89,6 GB/s** | Radeon 780M | OCuLink, 2× M.2, 2.5GbE | circa $320 barebone pagina AOOSTAR; **€450–750** a seconda RAM/SSD/import | [S][€][E] |
| **ThinkCentre M90q Gen 5 Tiny** | Intel 14ª gen, varianti i5/i7/i9, AVX2 | 2× SO-DIMM, documentazione tipica fino a 64 GB; 2×48 non certificato | 76,8–89,6 GB/s in base alla CPU | UHD/Iris Xe secondo CPU | M.2; riser PCIe opzionale su alcune configurazioni, alimentatore proprietario | usato/ricondizionato circa €500–1.000 | [S][€][E] |
| **Dell OptiPlex Micro 7020** | Core 14ª gen, AVX2 | 2× SO-DIMM, Dell dichiara massimo 64 GB, 4800/5600 | 76,8–89,6 GB/s | Intel UHD | M.2; nessun percorso eGPU adatto documentato | circa €400–800 usato/configurato | [S][€] |
| **HP Elite Mini 800 G9** | Core i5/i7/i9 12ª–14ª gen, AVX2 | 2× SO-DIMM DDR5-4800, HP dichiara 64 GB | **76,8 GB/s** | UHD/Iris Xe secondo CPU | M.2; nessun OCuLink | circa €350–750 ricondizionato | [S][€] |

**Prima correzione importante:** il fatto che due Mini PC abbiano “DDR5” non li rende equivalenti. Un MS-01 a DDR5-5200 ha un picco inferiore a un Ryzen a DDR5-5600; entrambi restano lontanissimi dai 692 GB/s nominali di due P40 e, soprattutto, non hanno una VRAM separata per gli strati accelerati. Il salto vero è il bus 256-bit/quad-channel di Strix Halo o la memoria unificata Apple a 400 GB/s.

### 2.2 Verifica specifica dei kit 2×48 GB

Il kit 2×48 GB DDR5 SO-DIMM non va dato per scontato:

| Famiglia | 2×48 GB | Giudizio operativo |
|---|---|---|
| MS-01 | documentato dal produttore come 96 GB DDR5-5200 e confermato da guide/upgrade | **Sufficientemente verificato [S]**, test memtest obbligatorio |
| UM780 XTX | pagina e rivenditori indicano 48 GB per slot/96 GB | **Verificato per la piattaforma [S/P]**, preferire kit JEDEC stabile |
| UM790 Pro | la pagina originale spesso si ferma a 64 GB, ma utenti hanno usato 96 GB | **Compatibile probabile [P], non garanzia del produttore** |
| UM890 Pro | produttore indica 96 GB | **Verificato [S]** |
| AI X1 Pro-370 | pagina globale/europea parla di 128 GB, ma il kit preciso 2×48 non è esplicitato in ogni variante BIOS | **Da provare [S/E]**; non pagare una variante 96 GB senza reso |
| SER8 | Beelink dichiara fino a 256 GB, ma la stabilità reale di 2×48 è dipendente da BIOS/kit | **Plausibile [S/E], da memtestare** |
| K8 Plus | pagina ufficiale indica fino a 96 GB | **Verificato [S]** |
| GEM12 Pro | pagina AOOSTAR indica fino a 128 GB | **Verificato come capacità dichiarata [S], kit specifico da testare** |
| Tiny aziendali 64 GB | Dell/HP/Lenovo dichiarano 64 GB su molte varianti | **Non promuovere 2×48** senza lista di compatibilità e reso |

Per il 70B la configurazione minima pratica non è “96 GB qualsiasi”: è un kit dual-channel stabile, con memoria non ridotta a 4800 MT/s dal BIOS, almeno 12–20 GB liberi dopo l’avvio e test di errore di memoria sotto carico iGPU. Una sola SO-DIMM dimezza la banda e può anche declassare la iGPU da Iris Xe a UHD sulle piattaforme Intel.

---

## 3. Analisi per CPU e architettura

### 3.1 Ryzen 7 7840HS, Ryzen 9 7940HS, Ryzen 7/9 8845HS/8945HS

Questi chip Phoenix/Hawk Point sono Zen 4 con 8 core e 16 thread, Radeon 780M e controller DDR5-5600/LPDDR5X-7500. AMD dichiara 8C/16T e 780M a 12 CU per il 7840HS; il 7940HS arriva a 5,2 GHz e la stessa classe di memoria. L’8845HS è un refresh con NPU, non un salto di banda o di numero di core. [S]

Per llama.cpp hanno due vantaggi rispetto agli Intel Raptor Lake mobili:

1. AVX-512 disponibile, normalmente eseguito su unità 256-bit in Zen 4; può aiutare il percorso CPU, ma non trasforma 89,6 GB/s in una GPU.
2. iGPU RDNA3 utilizzabile via Vulkan/ROCm, se driver e build sono corretti.

Il limite comune è la memoria: `89,6 GB/s` di picco è solo il 116,7% dello Xeon E5-2680 v4 da 76,8 GB/s e circa il 13% della banda nominale aggregata dual-P40. La CPU può avere latenza e IPC migliori dello Xeon, ma per un dense 70B quantizzato il decode è un flusso di lettura dei pesi; il vantaggio pratico resta nell’ordine di pochi tok/s.

**Previsione per 70B Q4:**

- CPU-only 7840HS/7940HS/8845HS/8945HS, 96 GB: **2,5–4,5 tok/s [E]**;
- offload parziale su 780M/Vulkan: **3–6 tok/s [E]**;
- OCuLink con una GPU discreta moderna e resto dei layer in RAM: **3–8 tok/s [E]**, dipendente dalla GPU, dal rapporto layer CPU/GPU e dal link.

Non esiste una misura pubblica omogenea Qwen2.5-72B GGUF su ciascuno di questi box. Un post sull’uso CPU di Qwen2.5-72B riporta tempi nell’ordine dei minuti, ma non è una piattaforma Mini PC controllata; un benchmark Geerling sul Minisforum MS-R1, che è hardware diverso, dà 0,77 tok/s su Llama 3.1 70B. Questi sono contesto, non valori da copiare sul 7840HS.

**Verdetto:** ottimi host economici e silenziosi, non macchine 70B chat promosse in CPU-only.

### 3.2 Core i9-13900H e Core Ultra 9 285HX

Il Core i9-13900H del MS-01 ha 6 P-core, 8 E-core, 20 thread, 45 W base power e fino a 115 W turbo; Intel dichiara DDR5-5200 e massimo 96 GB, due canali, AVX2 e Iris Xe fino a 1,5 GHz. [S]

Il Core Ultra 9 285HX dell’MS-02 è molto più interessante come piattaforma: 24 core/24 thread, 55 W base e 160 W maximum turbo, DDR5-6400, fino a 256 GB, ECC e 24 linee PCIe dichiarate. Non ha AVX-512: Intel riporta AVX2. [S] La banda sale a 102,4 GB/s, ma quattro SO-DIMM non sono quattro canali; con quattro moduli si riempiono due canali, aumentando capacità e potenzialmente rank interleaving, non il picco a 204,8 GB/s.

Per il solo percorso CPU, l’MS-02 può essere il più rapido fra i Mini PC x86 tradizionali, ma la potenza turbo e il raffreddamento sono molto superiori a quelli di un box da 35–65 W. È una workstation compatta, non un concorrente diretto di un SER8 silenzioso.

**Previsione 70B Q4:**

- MS-01 CPU-only: **2–4 tok/s [E]**;
- MS-02 285HX, memoria DDR5-6400 e profilo sostenuto: **4–7 tok/s [E]**;
- entrambi con una GPU discreta via PCIe e layer split ben configurato: possibile superamento di 8 tok/s, ma **non è una misura trovata [E]**.

Il MS-01 ha un percorso eGPU molto più utile del tipico Tiny: il produttore indica slot fisico x16 a velocità x8, oltre a 96 GB. Non va chiamato OCuLink interno; è una scheda PCIe a basso profilo o un adattatore per eGPU, con vincoli di alimentazione e ingombro. Il MS-02 ha un vero percorso PCIe 5.0 x16 e può diventare la base di una GPU discreta, ma a quel punto il progetto non è più “Mini PC CPU-only economico”.

### 3.3 Ryzen 9 9955HX e percorso AM5

Il MS-A2 usa Ryzen 9 9955HX Zen 5, 16 core/32 thread, Radeon 610M e memoria DDR5-5600 dual-channel fino a 96 GB. La pagina del produttore indica 89,6 GB/s e un PCIe fisico x16 elettrico x8, con split in due x4. [S]

È una distinzione fondamentale: 16 core Zen 5 possono migliorare il prefill e le operazioni CPU, ma il decode del 70B non scala con i core se la memoria resta a 89,6 GB/s. Il suo valore è l’espansione: una GPU discreta può fare il lavoro pesante, mentre il Mini PC fornisce RAM, storage, rete e un ambiente compatto.

L’MS-A1, con socket AM5 e Ryzen 7 8700G, è più riparabile e aggiornabile della maggior parte dei box mobili. L’8700G ha 8C/16T Zen 4 e Radeon 780M; il limite è ancora il dual-channel. Il socket consente di cambiare CPU e di costruire un percorso PCIe più flessibile, ma memoria, BIOS, alimentatore e supporto fisico vanno verificati per ogni combinazione. Il prezzo “barebone” è fuorviante: bisogna sommare CPU, 2×48 GB, SSD e, se serve, GPU/eGPU.

**Previsione 70B CPU-only:** MS-A1/8700G **3–5 tok/s [E]**; MS-A2 **4–7 tok/s [E]**. La promozione può avvenire solo con una GPU discreta e un test reale.

### 3.4 Ryzen AI 9 HX 370 e Radeon 890M

Il Ryzen AI 9 HX 370 ha 12 core/24 thread, architettura ibrida Zen 5/Zen 5c, boost fino a 5,1 GHz, Radeon 890M con 16 CU e supporto LPDDR5X-8000 o DDR5. AMD dichiara fino a 256 GB come limite del processore, ma il box decide se usare SO-DIMM, memoria saldata o una configurazione più piccola. [S]

Due box apparentemente simili hanno comportamenti diversi:

- **AI X1 Pro-370:** SO-DIMM, fino a 128 GB dichiarati, OCuLink; è il candidato HX370 pratico per 96 GB.
- **GMKtec EVO-X1:** tipicamente 32 GB LPDDR5X-7500 saldati, circa 120 GB/s, troppo pochi per Qwen 72B Q4 con margine. Non è corretto usare la specifica del processore per promuovere questa variante.

Su 2×DDR5-5600 il box HX370 ha lo stesso picco di un 7840HS, 89,6 GB/s, malgrado CPU e iGPU siano più nuove. L’890M può migliorare l’offload, ma il suo vantaggio è limitato dal bus. Su LPDDR5X-7500 saldata, la banda teorica sale a 120 GB/s, ma la RAM non è espandibile.

**Previsione:**

- AI X1 Pro 96 GB CPU-only: **3,5–5,5 tok/s [E]**;
- AI X1 Pro con 890M/Vulkan: **4–8 tok/s [E]**, borderline;
- EVO-X1 32 GB: **non fit robusto [S/C]**, da scartare per il target;
- 890M + OCuLink e GPU moderna: **potenzialmente ≥8 tok/s [E]**, ma il risultato dipende più dalla GPU e dallo split che dal nome HX370.

Sono stati trovati benchmark pubblici di 890M con modelli fino a circa 20B e numeri 15–27 tok/s, ma non sono benchmark Qwen2.5-72B/Llama 3.1 70B. Un dato su Qwen 14B mostra prompt speed elevato, ma anche questo non è decode 70B. [P] Non usare questi numeri per il business case.

### 3.5 Ryzen AI Max+ 395: il primo Mini PC con banda realmente diversa

Il Ryzen AI Max+ 395 ha 16 core/32 thread Zen 5, Radeon 8060S con 40 CU e una memoria LPDDR5X-8000 a 256 bit. AMD dichiara 128 GB massimi e il bus 256-bit; il calcolo è `8.000 MT/s × 32 byte = 256 GB/s`. [S/C]

Qui il confronto con i box dual-channel cambia:

| Memoria | Picco teorico | Rispetto a Xeon 76,8 | Rispetto a dual-P40 692 |
|---|---:|---:|---:|
| DDR5-5600 dual | 89,6 GB/s | 1,17× | 0,13× |
| LPDDR5X-8000 dual | 128 GB/s | 1,67× | 0,18× |
| Strix Halo 256-bit | **256 GB/s** | **3,33×** | **0,37×** |
| 2× P40 VRAM | 692 GB/s | 9,01× | 1× |
| M2 Max | 400 GB/s | 5,21× | 0,58× |

Il 395 può dedicare una regione configurabile della UMA all’iGPU e conservare il resto per CPU/OS. È una differenza architetturale reale, non marketing. La RAM però è saldata: l’acquisto da 64 GB è sbagliato per Qwen 72B Q4_K_M; la variante da 128 GB è quella da considerare.

**Benchmark pubblico rilevante:** il repository di Jeff Geerling riporta Llama 3.1 70B su un Framework Desktop con Ryzen AI Max+ 395 e 128 GB a **4,97 tok/s**, GPU/CPU, circa **133 W di picco**. [M] Il dato è sul modello esatto di classe 70B, ma non è un Mini PC Minisforum/GMKtec, non è Qwen2.5-72B e non dimostra il requisito di 8 tok/s.

AMD ha pubblicato una comunicazione secondo cui la piattaforma può eseguire LLM fino a 128B e raggiungere “fino a 15 tok/s”; è una cifra dichiarata in condizioni e modello non equivalenti, quindi [S/P], non un benchmark di accettazione. [S/P]

**Previsione prudente:** 5–8 tok/s su 70–72B Q4 [M/E], con 4,97 tok/s come ancora pubblica più importante; 8 tok/s è possibile ma non dimostrato. Per una chat prioritaria la variante 128 GB merita un test con reso, non una promozione automatica.

---

## 4. Via iGPU: quanto accelera davvero e quando è solo una parola

### 4.1 Radeon 780M

La Radeon 780M dei 7840HS/7940HS/8845HS ha 12 CU RDNA3. La sua memoria è la DDR5 del sistema: su 2×DDR5-5600 il picco è 89,6 GB/s condiviso. Il vantaggio rispetto alla CPU può derivare dalla parallelizzazione dei kernel e dall’offload di alcuni layer, ma il modello resta distribuito fra CPU e iGPU e ogni token può richiedere sincronizzazioni/copie.

Una configurazione con 96 GB permette di tenere il file da 40–42 GB, KV cache e sistema operativo; non significa che 96 GB diventino “96 GB di VRAM veloce”. Una configurazione da 64 GB può caricare IQ4_XS, ma ha poco margine per contesto e runtime. La Q4_K_M da circa 47 GB è sconsigliata su 64 GB.

**Stima chat 70B:**

- CPU-only: 2,5–4,5 tok/s [E];
- offload 780M Vulkan: 3–6 tok/s [E];
- con OCuLink e GPU discreta da 16 GB: 3–8 tok/s [E], da verificare.

Il requisito ≥8 tok/s non è superato sulla sola 780M. Il vantaggio dell’OCuLink è il path di crescita: può trasformare il box in una macchina 70B se si compra una GPU adeguata, ma il costo non è più quello del Mini PC.

### 4.2 Radeon 890M

L’890M ha 16 CU ed è più potente della 780M, ma su AI X1 Pro con DDR5-5600 condivide ancora circa 89,6 GB/s. Su EVO-X1 con LPDDR5X-7500 ha circa 120 GB/s, ma la variante da 32 GB non può contenere in modo robusto un 70B Q4. Il collo di bottiglia è quindi contemporaneamente banda e capacità.

La 890M è molto interessante per 7–20B, embedding e modelli multimodali più piccoli. Per 70B Q4 il risultato previsto di 4–8 tok/s è **borderline [E]**. Non va presentata come alternativa a due GPU discrete.

### 4.3 Radeon 8060S

La 8060S del 395 è la sola iGPU analizzata con 40 CU e bus 256-bit/256 GB/s. La capacità UMA di 128 GB risolve il fit del modello e lascia margine per KV cache. È tecnicamente la migliore soluzione compatta senza GPU discreta.

Ma anche qui il benchmark pubblico Llama 3.1 70B a 4,97 tok/s impedisce di dichiarare superata la soglia. Le cause possibili sono backend, allocazione UMA, versione di llama.cpp, quantizzazione e power limit; sono motivi per testare, non per sostituire il dato con un numero più favorevole.

### 4.4 Intel Arc/Iris nei Tiny

Il Core i9-13900H ha Iris Xe a 96 EU solo in dual-channel; i Core Ultra H possono avere Arc con Xe-core, ma il modello di iGPU e l’abilitazione dipendono dal sistema. Il Core Ultra 7 258V ha Arc 140V e 32 GB LPDDR5X-8533, circa 136,5 GB/s, ma la memoria è saldata e limitata a 32 GB: non fit per il 70B Q4. [S]

I Tiny business con Intel UHD/Iris Xe e massimo 64 GB possono eseguire versioni molto compresse o offload, ma non sono una scelta seria per il requisito. L’ecosistema llama.cpp SYCL/Vulkan è migliorato, però non c’è una misura pubblica omogenea sul target per questi chassis.

---

## 5. Schede per modello: cosa si compra realmente

### 5.1 Minisforum MS-01: CPU forte, piattaforma eGPU utile

**Componenti:** i9-13900H, due SO-DIMM DDR5-5200 fino a 96 GB, due porte USB4, tre M.2 e slot PCIe fisico x16 a x8. Minisforum dichiara compatibilità con una RTX A2000 Mobile e supporto Linux/Windows. [S]

**Punti forti:** 96 GB verificabili, rete 10GbE, storage numeroso, PCIe migliore di un box comune. **Punti deboli:** AVX2 senza AVX-512, memoria più lenta di Ryzen 5600, chassis e alimentatore non progettati per due GPU, ventola sotto carico. La Iris Xe non è un sostituto di una RTX.

**Uso consigliato:** host per una GPU discreta single-slot/low-profile o server di storage, non CPU-only 70B. Con GPU aggiunta il prezzo iniziale di circa €709 non è più comparabile con UM780 da €350.

### 5.2 UM780 XTX e UM890 Pro: miglior base economica OCuLink

UM780 XTX è la scelta economica se si trova il refurb a circa €349. UM890 Pro è più recente, con 8945HS e fino a 96 GB ufficiali, a circa €489 con 32 GB/1 TB in uno snapshot EU. Entrambi hanno 780M, due SO-DIMM e OCuLink.

Il 8945HS non raddoppia la banda né i core rispetto al 7840HS; il maggior valore è il supporto, la disponibilità e la porta. Per CPU-only sono equivalenti in ordine di grandezza. L’UM780 ha senso se:

- si possiede già una GPU esterna;
- si vuole spendere poco per l’host;
- il 70B non deve necessariamente superare 8 tok/s senza GPU.

### 5.3 UM790 Pro: prezzo interessante, ma attenzione alla capacità ufficiale

Il 7940HS è un buon Zen 4, ma la pagina ufficiale consultata riporta varianti fino a 64 GB mentre i report utenti mostrano 96 GB. È un caso in cui la verifica del kit è parte del prezzo: comprare 64 GB e scoprire che 2×48 non fa boot annulla il vantaggio.

Senza OCuLink, una eGPU USB4 ha più overhead e meno banda utile. Va scelto solo se è molto meno costoso di UM890 o se si intende usare una GPU via USB4 per modelli inferiori.

### 5.4 MS-A1 e MS-A2: workstation upgradeabili, non Mini PC a basso costo

MS-A1 è il percorso più riparabile: socket AM5, CPU sostituibile e componenti desktop. È interessante per chi vuole iniziare con 8700G e aggiungere una GPU. Il costo barebone non include la parte che determina la riuscita del progetto: CPU, RAM, SSD, eventuale scheda video, riser e alimentazione.

MS-A2 con 9955HX è più potente lato CPU e offre PCIe x8, tre M.2, U.2 e due SO-DIMM fino a 96 GB. La Radeon 610M non ha valore per accelerare un 70B. Il modello va valutato come “host compatto per GPU”, non come soluzione CPU-only. Se il budget consente MS-A2 + GPU, bisogna confrontarlo direttamente con una workstation desktop o dual-3090, non con un SER8.

### 5.5 AI X1 Pro-370 ed EVO-X1: stesso nome AI, due prodotti diversi

AI X1 Pro-370 è quello sensato per questo target perché può arrivare a 96/128 GB, ha OCuLink e RAM SO-DIMM. EVO-X1 è più piccolo e usa spesso 32 GB LPDDR5X saldati: ottimo per 7–20B, **non il box da comprare per Qwen 72B**.

Il prezzo AI X1 Pro configurato con 64 GB è già circa €1.639 nello snapshot ufficiale EU; la variante 96 GB può salire a circa €1.800–2.000. A quel livello un Mac Studio usato o un host con eGPU deve essere incluso nel confronto.

### 5.6 MS-S1 MAX ed EVO-X2: memoria giusta, prezzo alto e benchmark ancora insufficiente

MS-S1 MAX ed EVO-X2 sono tecnicamente i candidati più interessanti: 128 GB UMA, 256 GB/s, 40 CU e 16 core Zen 5. Il prezzo però è circa €3.2–4.0k in EU, con RAM non sostituibile. Il loro acquisto può essere razionale se il valore è avere un box silenzioso che esegue anche modelli da 100–128B e se il test reale conferma 8 tok/s.

Il benchmark Framework/Geerling da 4,97 tok/s e il prezzo alto impediscono di chiamarli “best buy” per il solo Qwen 72B. Se il solo obiettivo è 70B a 8+ tok/s, la dual-3090 o una GPU cloud possono costare meno; se il requisito è 128 GB UMA silenziosa e molti modelli locali, Strix Halo è l’unica strada compatta da provare.

### 5.7 Beelink, GMKtec e AOOSTAR

- **Beelink SER8:** silenzioso, 8845HS, 2×SO-DIMM e 2×M.2. La promessa di 256 GB è interessante, ma il kit 2×48 e il mantenimento della frequenza devono essere testati. Non ha OCuLink documentato nella pagina consultata; quindi è meno adatto dell’UM780/K8 come host eGPU.
- **GMKtec K8 Plus:** 8845HS, 96 GB dichiarati, OCuLink e prezzo da circa $400. È la migliore alternativa economica a UM780/UM890 se il venditore garantisce 2×48 GB e il reso. Non cambia il verdetto CPU-only.
- **AOOSTAR GEM12 Pro:** 8845HS, 2×SO-DIMM fino a 128 GB, OCuLink e due M.2. Il prezzo barebone di circa $320 è attraente, ma la spedizione, IVA e assistenza riportano spesso il sistema a €450–750. La variante è da testare come host OCuLink, non da promuovere come 70B CPU-only.
- **Acemagic/Bosgame e cloni:** possono usare gli stessi SoC, ma BIOS, RAM, raffreddamento e garanzia differiscono. Senza una pagina tecnica che confermi due slot, kit 2×48, TDP sostenuto e reso europeo, non hanno un vantaggio dimostrato sul K8/UM890.

---

## 6. Benchmark e stime per modello

### 6.1 Misure pubblicate pertinenti

| Misura | Risultato | Cosa dimostra | Cosa non dimostra |
|---|---:|---|---|
| Llama 3.1 70B, M3 Ultra 512 GB, Geerling | 14,08 tok/s, 243 W peak | un’architettura Apple con molta banda può superare 8 | non è M2 Max né Mini PC x86 |
| Llama 3.1 70B, M1 Ultra 128 GB, Geerling | 9,84 tok/s | proxy Apple appena sopra soglia | non è M2, né Qwen, né box x86 |
| Llama 3.1 70B, M1 Max 64 GB, Geerling | 7,25 tok/s | una soluzione Apple compatta può restare sotto 8 | non generalizzare a M2/M4 |
| Llama 3.1 70B, Framework/395+ 128 GB, Geerling | **4,97 tok/s, 133 W** | benchmark direttamente rilevante per Strix Halo | non è MS-S1/EVO-X2; runtime e firmware possono differire |
| Llama 3.1 70B, Minisforum MS-R1 CPU | 0,77 tok/s, 38,2 W | CPU-only può essere molto lento | MS-R1 non è 7840HS/8845HS |
| Qwen2.5-72B GPTQ-Int4, A100 singola, Qwen | 11,07 tok/s | GPU datacenter supera la soglia a contesto corto | non è Mini PC, formato GGUF diverso |
| Qwen2.5-72B GPTQ-Int4, vLLM A100 | 16,47 tok/s | serving moderno dà margine | non trasferibile a RAM DDR5 |
| Qwen2.5-72B, 2×A100 vLLM | 46,30 tok/s | il parallelismo GPU è su un altro ordine | costo e infrastruttura completamente diversi |

Fonte primaria dei dati Geerling: repository e issue del Framework Desktop. La sua metodologia avverte correttamente che modello, quantizzazione, contesto, backend e distinzione prompt/eval cambiano il risultato. Il numero 4,97 tok/s è **decode/eval** e quindi più utile per questa decisione di un numero di prompt processing.

### 6.2 Previsione per ogni candidato

La tabella seguente è una stima condizionale, non un benchmark. Assume file Qwen2.5-72B IQ4_XS/Q4_0, contesto 4k, sistema caldo e llama.cpp correttamente configurato.

| Modello | CPU-only 70B Q4 | iGPU/offload interno | Con eGPU/GPU aggiunta | Chat ≥8? |
|---|---:|---:|---:|---|
| MS-01 i9-13900H/96 | 2–4 [E] | 2–4 [E] Iris Xe | 5–10 [E], dipende da GPU e x8 | No senza test/GPU |
| UM780 XTX/96 | 2,5–4 [E] | 3–6 [E] 780M | 3–8 [E] OCuLink | No / borderline |
| UM790 Pro/96 | 2,5–4,5 [E] | 3–6 [E] 780M | 3–7 [E] USB4 | No |
| UM890 Pro/96 | 3–4,5 [E] | 3–6 [E] 780M | 4–8 [E] OCuLink | No / borderline |
| MS-A1 + 8700G/96 | 3–5 [E] | 3–6 [E] 780M | 6–12 [E] PCIe desktop | Solo con GPU e test |
| MS-A2 9955HX/96 | 4–7 [E] | 4–6 [E] 610M | 7–12 [E] PCIe x8 | Solo con GPU e test |
| MS-02 285HX/96–256 | 4–7 [E] | 3–5 [E] Intel Graphics | 8+ possibile [E] PCIe x16 | Solo con GPU e test |
| AI X1 Pro-370/96 | 3,5–5,5 [E] | 4–8 [E] 890M | 5–10 [E] OCuLink | Non certificata |
| SER8/96 | 2,5–4 [E] | 3–6 [E] 780M | 3–7 [E] USB4 | No |
| K8 Plus/96 | 2,5–4 [E] | 3–6 [E] 780M | 3–8 [E] OCuLink | No / borderline |
| GEM12 Pro/96–128 | 2,5–4 [E] | 3–6 [E] 780M | 3–8 [E] OCuLink | No / borderline |
| EVO-X1/32 | non fit robusto [S/C] | non fit robusto | possibile solo con eGPU e RAM insufficiente | No |
| **MS-S1 MAX/128** | 4,5–7 [E] | **4,97 misurati su proxy 395 [M]** | 5–10 [E], PCIe x4 | **Non ancora** |
| **EVO-X2/128** | 4,5–7 [E] | 4,97 proxy 395 [M] | 5–10 [E] | Non ancora |
| M2 Max/96, comparatore | N/D | circa 7–10 [P/E] | — | Test obbligatorio |

Un valore centrale più alto degli intervalli non è autorizzato dal marketing. In particolare, “Radeon 8060S come RTX 4070 Laptop” è un confronto gaming/teorico e non un dato di decode LLM.

### 6.3 Chat contro asincrono

In chat conta il tempo fra token; in asincrono conta il throughput totale. Un Mini PC con 4 tok/s può completare un job notturno senza problemi, ma fallisce il requisito chat. Un iGPU con offload può migliorare il prompt processing e non il decode: è un errore comune usare il primo numero per vendere il secondo.

Per il carico misto si deve testare una chat mentre un job da 2.000 token è attivo. La 780M/890M con RAM contesa è particolarmente sensibile; Strix Halo ha più banda ma anche l’iGPU e CPU condividono la stessa UMA. La soglia di accettazione resta 8 tok/s durante la contesa, non solo a macchina inattiva.

---

## 7. Prezzi, componenti e configurazioni acquistabili

### 7.1 Stima della distinta base

| Configurazione | Box/host | RAM + SSD aggiunti | eGPU/GPU | Capex realistico EU |
|---|---:|---:|---:|---:|
| UM780 refurb + 96 GB + 1 TB | €349 | €220–300 | — | **€570–700** |
| UM890 + 96 GB + 1 TB | €489–600 | già/extra €150–250 | — | **€650–850** |
| K8 Plus + 96 GB + 1 TB | €400–550 | €180–250 | — | **€600–800** |
| GEM12 Pro + 96/128 GB + 1 TB | €450–600 | spesso incluso/€150 | — | **€600–850** |
| MS-01 + 96 GB + 1 TB | €709 circa | spesso incluso, altrimenti €200 | — | **€800–1.000** |
| AI X1 Pro + 96 GB + 1 TB | €729 barebone | €300–500 | — | **€1.100–1.900** |
| MS-A2 + 96 GB + SSD | €839 base | €250–450 | — | **€1.100–1.500** |
| Host OCuLink + 4060 Ti 16 GB + dock/PSU | sopra host | — | €350–550 usata/nuova, dock €100–250 | **€1.000–1.600** |
| MS-S1 MAX 128 GB | — | saldata | inclusa | **€3.300–4.000** |
| EVO-X2 128 GB | — | saldata | inclusa | **€3.200–3.500** |
| SER8 64/96 GB + SSD | circa €650–800 | incluso/extra | — | **€700–950** |

I prezzi USA in dollari non vanno convertiti a euro senza IVA, spedizione e garanzia. Un prezzo Taobao/AliExpress può essere più basso, ma occorre aggiungere il 22% IVA italiana, eventuale dazio/commissione, spedizione, alimentatore, reso internazionale e rischio BIOS. Una configurazione “barebone” non è più economica se RAM e SSD locali costano quanto la differenza rispetto alla variante completa.

### 7.2 Miglior acquisto per obiettivo

- **Costo minimo per host 70B/eGPU:** UM780 XTX refurb o K8 Plus, se 2×48 GB e OCuLink sono garantiti.
- **Host più completo e robusto:** MS-01 o MS-A2; il PCIe è migliore, ma capex e rumore aumentano.
- **70B senza GPU discreta:** MS-S1 MAX/EVO-X2 128 GB è l’unica via Mini PC tecnicamente seria, ma il prezzo è alto e la misura 4,97 tok/s non passa la soglia.
- **Box da evitare per questo target:** SER8 32 GB, EVO-X1 32 GB, Tiny 64 GB, UM790 con RAM non verificata.
- **Alternativa fuori categoria ma economicamente concreta:** una RTX 3090 usata singola per modelli più piccoli + cloud A100/L40S per il 70B, oppure dual-3090 se rumore e 700–800 W sono accettabili.

---

## 8. TCO a tre anni a 16 ore/giorno

### 8.1 Ipotesi

Uso un caso base identico all’analisi precedente:

```text
16 h/giorno × 360 giorni/anno = 5.760 h/anno
energia = potenza media alla presa × 5.760 h × 0,30 €/kWh
fondo manutenzione = 5% del capex per anno
TCO 3y = capex + 3 × energia annua + 15% × capex
      = 1,15 × capex + 5,184 × potenza media in watt
```

La potenza è media alla presa durante l’uso, non TDP. È una stima [E]. Il Mini PC CPU-only in realtà può consumare meno quando non genera; qui si assume uso attivo 16 ore/giorno. Il costo della climatizzazione non è incluso: per i box da 130–200 W può essere significativo in estate.

### 8.2 Potenza e TCO per candidato

| Candidato configurato | Capex usato | Potenza media stimata | Energia annua a €0,30 | TCO 3 anni | Fit 70B | Evidenza |
|---|---:|---:|---:|---:|---|---|
| UM780 XTX/96 CPU-only | €650 | 50 W | €86 | **€1.007** | sì, margine | [€][E] |
| UM790 Pro/96 CPU-only | €750 | 55 W | €95 | **€1.148** | sì, ma RAM da provare | [€][E] |
| UM890 Pro/96 CPU-only | €800 | 55 W | €95 | **€1.205** | sì | [€][E] |
| K8 Plus/96 CPU-only | €650 | 55 W | €95 | **€1.033** | sì | [€][E] |
| GEM12 Pro/128 CPU-only | €800 | 60 W | €104 | **€1.231** | sì | [€][E] |
| SER8/96 CPU-only | €850 | 55 W | €95 | **€1.262** | da verificare | [€][E] |
| MS-01/96 CPU-only | €900 | 60 W | €104 | **€1.346** | sì | [€][E] |
| MS-A1/8700G/96 | €900 | 75 W | €130 | **€1.424** | sì | [€][E] |
| MS-A2/9955HX/96 | €1.100 | 75 W | €130 | **€1.654** | sì | [€][E] |
| MS-02/285HX/96 | €1.400 | 90 W | €156 | **€2.077** | sì | [€][E] |
| AI X1 Pro/96 | €1.900 | 70 W | €121 | **€2.548** | sì | [€][E] |
| MS-S1 MAX/128 | €3.500 | 140 W | €242 | **€4.751** | sì con grande margine memoria | [€][E] |
| EVO-X2/128 | €3.230 | 140 W | €242 | **€4.440** | sì | [€][E] |
| EVO-X1/32 | €1.250 | 65 W | €112 | **€1.775** | no robusto | [€][E] |
| M2 Max/96 comparatore | €1.500 | 80 W | €138 | **€2.140** | sì | [€][P/E] |

Le potenze Strix Halo variano con il profilo: Minisforum dichiara 130 W sostenuti/160 W peak per MS-S1, quindi 140 W medi è un’ipotesi attiva, non una misura. Per i box da 45–65 W il wattaggio del SoC non è il wattaggio alla presa; alimentatore, SSD e display possono modificarlo.

### 8.3 Sensibilità elettrica

Il costo triennale dell’energia per ogni 10 W medi aggiuntivi è:

- a €0,25/kWh: `10 × 5.760 × 3 × 0,25/1000 = €43,20`;
- a €0,30/kWh: **€51,84**;
- a €0,35/kWh: `€60,48`.

| Potenza | TCO energia 3 anni a €0,25 | a €0,30 | a €0,35 |
|---:|---:|---:|---:|
| 50 W | €216 | €259 | €302 |
| 60 W | €259 | €311 | €363 |
| 75 W | €324 | €389 | €454 |
| 90 W | €389 | €467 | €544 |
| 140 W | €605 | €726 | €847 |

Il vantaggio energetico di un Mini PC da 50 W rispetto a una dual-3090 è reale, ma il costo energia non risolve il problema prestazionale. Se il box produce 3 tok/s e un’altra architettura 12 tok/s, il confronto corretto è energia per token utile e tempo dell’utente, non solo euro/anno.

### 8.4 TCO e prezzo cloud

Un A100 on-demand a circa €1,25/GPU·h con 10% di overhead e 120 €/anno storage costa circa:

- 80 h/mese: €1.440/anno;
- 160 h/mese: €2.760/anno;
- 480 h/mese: €8.040/anno.

Il Mini PC da €650–1.200 ha un TCO triennale inferiore a €1.700, ma non offre il throughput di un’A100. Il Mini PC Strix Halo da €3.2–4k ha TCO triennale simile o superiore a una buona soluzione Apple usata, pur avendo un benchmark 70B pubblico sotto 8 tok/s. Il cloud resta economicamente preferibile quando l’uso è intermittente e il requisito è velocità garantita; il locale diventa preferibile solo se il dispositivo è utilizzato molte ore, la privacy è obbligatoria o il modello deve essere sempre disponibile.

---

## 9. Matrice decisionale pesata

### 9.1 Pesi e regola anti-overclaim

| Criterio | Peso |
|---|---:|
| Chat: decode + TTFT | 30% |
| Asincrono/batch | 15% |
| Silenzio e calore | 15% |
| Consumo | 15% |
| TCO 3 anni | 15% |
| Upgrade/manutenzione | 5% |
| Futuro/supporto software | 5% |

Punteggio da 1 a 5. Un benchmark mancante sul modello esatto limita il punteggio chat a 3 anche se la teoria è favorevole. Il totale è la somma `peso × voto/5`.

### 9.2 Matrice Mini PC e percorsi collegati

| Path | Chat | Async | Silenzio | Energia | TCO | Upgrade | Futuro | Totale /100 | Esito |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| UM780/K8/GEM12 CPU-only | 1 | 1 | 5 | 5 | 5 | 4 | 3 | **61** | scarta per chat |
| UM890 CPU-only | 1 | 1 | 5 | 5 | 5 | 4 | 3 | **61** | scarta per chat |
| SER8 CPU/iGPU | 1 | 1 | 5 | 5 | 4 | 4 | 3 | **60** | scarta |
| MS-01 CPU-only | 1 | 1 | 3 | 4 | 4 | 5 | 4 | **51** | host, non 70B |
| MS-A1 + GPU futura | 3 | 2 | 3 | 3 | 3 | 5 | 4 | **60** | test con GPU |
| MS-A2 + GPU futura | 3 | 3 | 3 | 3 | 3 | 4 | 4 | **62** | test con GPU |
| MS-02 + GPU PCIe x16 | 3 | 3 | 2 | 2 | 2 | 5 | 5 | **55** | workstation, non silenziosa |
| AI X1 Pro/96 iGPU | 2 | 2 | 4 | 4 | 3 | 4 | 4 | **59** | test solo con reso |
| **MS-S1 MAX/128** | 2 | 2 | 4 | 4 | 2 | 2 | 4 | **54** | test obbligatorio |
| **EVO-X2/128** | 2 | 2 | 4 | 4 | 2 | 2 | 4 | **55** | test obbligatorio |
| EVO-X1/32 | 1 | 1 | 4 | 4 | 2 | 1 | 4 | **44** | scarta: memoria |
| M2 Max/96 | 3 | 2 | 5 | 5 | 4 | 2 | 4 | **72** | test obbligatorio |
| Dual-RTX 3090 | 4 | 4 | 1 | 1 | 2 | 2 | 4 | **54** | supera chat possibile, non silenzio |
| Cloud A100 caldo | 5 | 5 | 5 | 5 | 1 a uso continuo | 4 | 5 | **87 intermittente** | riferimento qualità |

Il punteggio basso dello Strix Halo non significa che il prodotto sia scarso: riflette prezzo elevato, RAM non sostituibile e benchmark 4,97 tok/s. Il punteggio alto dei Mini PC CPU-only riflette silenzio/TCO, ma non autorizza a chiamarli adatti alla chat 70B. La matrice è decisionale, non una classifica sintetica di potenza.

---

## 10. Test di accettazione e protocollo di acquisto

### 10.1 Test comune

1. Usare Qwen2.5-72B-Instruct-IQ4_XS o Q4_0 con hash SHA-256 registrato; ripetere con Llama 3.1 70B Q4_K_M se la memoria lo permette.
2. Contesto 4.096, prompt 512 token, output 1.000 token, un solo slot.
3. Runtime e commit: llama.cpp recente con backend esplicitato (CPU, Vulkan, ROCm, Metal, CUDA).
4. Tre run dopo warm-up; riportare decode/eval, prompt processing, TTFT P50/P95 e memoria.
5. Ripetere a contesto 8.192 e con un job asincrono da 2.000 token.
6. Watt alla presa e temperatura dopo 30–60 minuti; misurare rumore a 50 cm in idle e carico.

**Promozione chat:** decode medio ≥8 tok/s, nessun calo sotto 6 tok/s per più di 10 secondi, TTFT caldo P95 ≤2 s sul prompt 512 e ≤5 s a 4k, memoria libera ≥8 GB dopo caricamento, nessun errore Vulkan/ROCm/driver.

### 10.2 Test specifico per RAM

- installare 2×48 GB dello stesso kit, non un modulo da 96 GB;
- verificare che il BIOS mantenga 5600/6400 MT/s e dual-channel;
- eseguire Memtest86 o stress test Linux per almeno una notte;
- misurare la banda con STREAM o benchmark equivalente, ma separare banda CPU e banda iGPU;
- in Strix Halo verificare la quota UMA assegnata e che 128 GB restino disponibili al modello.

### 10.3 Test specifico OCuLink/eGPU

- verificare che il link negozi PCIe 4.0×4, non Gen3 o fallback USB4;
- usare una GPU alimentata da PSU adeguato e non dal piccolo alimentatore del box;
- misurare il decode con 0, metà e tutti i layer offloadati;
- riportare interconnessione, trasferimenti per token e comportamento a contesto 8k;
- non definire “successo” un incremento del prompt processing senza incremento del decode.

### 10.4 Diritti di reso e rischio prezzi

Per AI X1 Pro, MS-S1, EVO-X2 e box importati il diritto di reso è parte del prezzo. Un dispositivo da €3.500 senza reso non è comparabile con uno da €1.500 acquistabile con restituzione. Prima dell’acquisto chiedere per iscritto:

- supporto reale di 2×48 GB;
- limite di potenza sostenuto per 30 minuti;
- compatibilità Linux/ROCm/Vulkan;
- versione BIOS e possibilità di aggiornamento;
- rumore dichiarato o procedura di restituzione;
- IVA, indirizzo di reso e durata garanzia.

---

## 11. Shortlist finale: testare, condizionare, scartare

### 11.1 Candidati che meritano un test reale

**1. MS-S1 MAX 128 GB / EVO-X2 128 GB — priorità tecnica, non economica.**  
Sono gli unici Mini PC analizzati con 128 GB UMA e 256 GB/s. Testarli solo con reso, perché l’unico benchmark pubblico pertinente (Framework 395, Llama 3.1 70B) è 4,97 tok/s. Promuoverli soltanto se il file Qwen scelto supera 8 tok/s e il rumore è accettabile. Il prezzo EU sopra €3.000 è giustificato solo da modelli oltre 70B, privacy e uso intensivo.

**2. AI X1 Pro-370 96 GB — priorità rapporto compattezza/eGPU.**  
12C/24T, 890M, OCuLink e SO-DIMM. La banda CPU è solo 89,6 GB/s, quindi il percorso interno è borderline; merita un test se il prezzo 96 GB resta vicino a €1.5–1.8k e se la eGPU è già disponibile.

**3. UM780 XTX, UM890 Pro, GMK K8 Plus, AOOSTAR GEM12 Pro — priorità prezzo/host.**  
Scegliere il più economico con 96 GB e OCuLink garantiti. Sono buoni nodi host per una futura GPU e macchine per modelli 7–32B. Non comprarli aspettandosi 8 tok/s CPU-only sul 70B.

**4. MS-A2 9955HX e MS-A1 — priorità upgrade.**  
Testarli soltanto se il progetto include una GPU PCIe discreta. Il vantaggio è la manutenzione/espansione; il prezzo del barebone non è il prezzo del sistema funzionante.

**5. MS-01/MS-02 — priorità storage, rete e GPU PCIe.**  
MS-01 è un host compatto interessante a x8; MS-02 è la workstation più espandibile e potente, ma più costosa e meno silenziosa. Nessuno dei due è promosso CPU-only.

### 11.2 Strade da scartare per il target dichiarato

- **EVO-X1 32 GB:** non fit robusto del file 70B Q4; il nome “AI” e la 890M non cambiano la capacità.
- **Beelink SER8 24/32 GB:** ottimo Mini PC generale, non macchina 70B. La variante 96 GB va comunque testata e non ha OCuLink documentato.
- **Tiny business HP/Dell/Lenovo da 64 GB:** silenziosi ed economici, ma capacità e banda limitano il 70B; riser/alimentatore eGPU sono troppo specifici.
- **UM790 Pro 64 GB senza kit/resi:** non pagare la promessa di 96 GB non formalizzata dal produttore.
- **Qualunque 780M/890M venduta con “VRAM condivisa”:** non trattarla come VRAM GDDR/HBM; è RAM condivisa a 89,6–120 GB/s.

### 11.3 Decisione finale condizionale

| Condizione dell’utente | Decisione |
|---|---|
| Obiettivo assoluto: ≥8 tok/s, silenzio, budget <€2.000 | non Mini PC CPU-only; GPU locale moderna + cloud 70B, oppure Mac usato testato |
| Uso 70B raro/intermittente | Mini PC economico per modelli piccoli + A100/L40S on-demand |
| Uso 70B locale 16h/giorno, silenzio prioritario | provare MS-S1/EVO-X2 128 GB e Mac Studio 64/96; comprare solo sopra soglia |
| Uso 70B locale 16h/giorno, velocità prioritaria | dual-3090 o GPU discreta in MS-A2/MS-02; accettare rumore/consumi |
| Budget host €600–900 e GPU futura | UM780 XTX/UM890/K8/GEM12 con 96 GB e OCuLink |
| Necessità di RAM >96 GB e crescita | MS-02 285HX o Strix Halo; il primo richiede GPU per chat, il secondo è saldato |
| Privacy offline e modelli 70–128B, velocità ancora da testare | Strix Halo 128 GB; rischio prezzo e benchmark esplicito |

**Verdetto:** il Mini PC non va respinto come categoria. Va diviso in tre componenti decisivi:

1. **dual-channel DDR5 89,6 GB/s:** economico, silenzioso, ma insufficiente per la chat 70B CPU-only;
2. **iGPU 780M/890M con RAM condivisa:** utile per modelli più piccoli e offload sperimentale, non equivalente a una GPU discreta;
3. **Strix Halo 256 GB/s + 128 GB UMA:** unica variante compatta che merita un test 70B senza eGPU, ma la misura pubblica 4,97 tok/s non passa ancora la soglia.

La scelta più prudente resta acquistare un host Mini PC economico solo se serve comunque per modelli piccoli, e non confondere il suo TCO basso con il superamento della soglia chat. Per l’obiettivo preciso dell’utente, la raccomandazione non cambia: **test Strix Halo con reso se silenzio/locale sono prioritari; altrimenti GPU discreta moderna o cloud ibrido.**

---

## 12. Fonti numerate

### Specifiche CPU, memoria e piattaforme

1. AMD, Ryzen 7 7840HS — specifiche DDR5-5600/LPDDR5X-7500 e Radeon 780M:  
   https://www.amd.com/en/products/processors/laptop/ryzen/7000-series/amd-ryzen-7-7840hs.html
2. AMD, Ryzen 9 7940HS:  
   https://www.amd.com/en/products/processors/laptop/ryzen/7000-series/amd-ryzen-9-7940hs.html
3. AMD, Ryzen 7 8845HS:  
   https://www.amd.com/en/products/processors/laptop/ryzen/8000-series/amd-ryzen-7-8845hs.html
4. AMD, Ryzen AI 9 HX 370:  
   https://www.amd.com/en/products/processors/laptop/ryzen/ai-300-series/amd-ryzen-ai-9-hx-370.html
5. AMD, Ryzen AI Max+ 395:  
   https://www.amd.com/en/products/processors/laptop/ryzen/ai-300-series/amd-ryzen-ai-max-plus-395.html
6. Intel, Core i9-13900H:  
   https://www.intel.com/content/www/us/en/products/sku/232135/intel-core-i913900h-processor-24m-cache-up-to-5-40-ghz/specifications.html
7. Intel, Core Ultra 9 285HX:  
   https://www.intel.com/content/www/us/en/products/sku/242297/intel-core-ultra-9-processor-285hx-36m-cache-up-to-5-50-ghz/specifications.html
8. Intel, Core Ultra 7 155H:  
   https://www.intel.com/content/www/us/en/products/sku/236847/intel-core-ultra-7-processor-155h-24m-cache-up-to-4-80-ghz/specifications.html
9. Intel, Core Ultra 7 258V:  
   https://www.intel.com/content/www/us/en/products/sku/240957/intel-core-ultra-7-processor-258v-12m-cache-up-to-4-80-ghz/specifications.html

### Mini PC e workstation compatte

10. Minisforum MS-01 — CPU, 96 GB, M.2 e PCIe 4.0 x8:  
    https://store.minisforum.com/products/minisforum-ms-01-workstation
11. Minisforum UM780 XTX:  
    https://www.minisforum.com/products/elitemini-um780-xtx
12. Minisforum UM790 Pro:  
    https://store.minisforum.com/products/minisforum-um790-pro-mini-pc
13. Minisforum UM890 Pro:  
    https://store.minisforum.com/products/minisforum-um890pro-mini-pc
14. Minisforum MS-A1, socket AM5:  
    https://www.minisforum.com/products/minisforum-ms-a1
15. Minisforum MS-A2 — 9955HX, 96 GB, PCIe x8 split, M.2/U.2:  
    https://store.minisforum.com/products/minisforum-ms-a2-workstation
16. Minisforum MS-S1 MAX — 395, 128 GB LPDDR5X-8000, 256-bit, 8060S, potenza e PCIe:  
    https://store.minisforum.com/products/minisforum-ms-s1-max-mini-pc
17. Beelink SER8 — 8845HS, DDR5-5600, capacità dichiarata, 2×M.2:  
    https://www.bee-link.com/products/beelink-ser8-8845hs
18. GMKtec K8 Plus — 8845HS e capacità fino a 96 GB:  
    https://www.gmktec.com/products/gmktec-nucbox-k8-plus-mini-pc-amd-ryzen-7-8845hs
19. GMKtec EVO-X1 — HX370/890M/LPDDR5X e prezzo EU:  
    https://de.gmktec.com/en/products/gmktec-evo-x1-amd-ryzen-ai-9-hx-370
20. GMKtec EVO-X2 — Ryzen AI Max+ 395 e varianti 64/128 GB:  
    https://de.gmktec.com/en/products/gmktec-evo-x2-amd-ryzen-ai-max-395-mini-pc-1
21. AOOSTAR GEM12/GEM12 Pro — 8845HS, 2×SO-DIMM, OCuLink:  
    https://aoostar.com/products/aoostar-gem12-amd-r7-pro-8845hs-mini-pc
22. Lenovo PSREF ThinkCentre M90q Gen 5:  
    https://psref.lenovo.com/Product/ThinkCentre_M90q_Gen_5?tab=spec
23. Dell OptiPlex Micro 7020, manuale memoria:  
    https://www.dell.com/support/manuals/en-us/optiplex-7020-micro/optiplex-micro-7020-owners-manual/memory
24. HP Elite Mini 800 G9, memoria e slot:  
    https://support.hp.com/us-en/document/ish_5868444-5868508-16

### Benchmark LLM e backend

25. Jeff Geerling, repository AI/LLM benchmarks con Llama 3.1 70B su M1/M3, Framework 395 e MS-R1:  
    https://github.com/geerlingguy/ai-benchmarks
26. Jeff Geerling, benchmark Framework Desktop/Ryzen AI Max+ 395, 128 GB:  
    https://github.com/geerlingguy/ai-benchmarks/issues/21
27. Qwen, Speed Benchmark ufficiale Qwen2.5-72B su A100 con GPTQ/AWQ/vLLM:  
    https://qwen.readthedocs.io/en/v2.5/benchmark/speed_benchmark.html
28. AMD, comunicazione su Ryzen AI Max e LLM fino a 128B; dato “fino a 15 tok/s”, da trattare come claim non benchmark comparabile:  
    https://www.amd.com/en/blogs/2025/amd-ryzen-ai-max-upgraded-run-up-to-128-billion-parameter-llms-lm-studio.html
29. Framework Community, test LLM su Strix Halo/Ryzen AI Max+ 395:  
    https://community.frame.work/t/amd-strix-halo-ryzen-ai-max-395-gpu-llm-performance-tests/72521
30. llama.cpp, benchmark database/progetto utilizzato per riproduzione:  
    https://github.com/ggml-org/llama.cpp

### Prezzi e mercato citati

31. Minisforum EU MS-01, snapshot del prezzo del box e condizioni di garanzia:  
    https://minisforumpc.eu/products/ms-01
32. Minisforum EU AI X1 Pro-370, configurazioni 32/64 GB e prezzi:  
    https://minisforumpc.eu/products/ai-x1-pro-mini-pc
33. Minisforum EU MS-S1 MAX 64/128 GB:  
    https://minisforumpc.eu/products/minisforum-ms-s1-max-mini-pc
34. GMKtec EU EVO-X1, configurazione e prezzo warehouse EU:  
    https://de.gmktec.com/products/gmktec-evo-x1-amd-ryzen-ai-9-hx-370
35. GMKtec EU EVO-X2, configurazioni 128 GB:  
    https://de.gmktec.com/en/products/gmktec-evo-x2-amd-ryzen-ai-max-395-mini-pc-1
36. Minisforum EU UM780 XTX refurb:  
    https://minisforumpc.eu/products/um780-xtx
37. Minisforum EU UM890 Pro:  
    https://minisforumpc.eu/products/minisforum-um890-pro-mini-pc
38. Beelink SER8, pagina ufficiale con prezzo e variante 32/64 GB:  
    https://www.bee-link.com/products/beelink-ser8-8845hs

---

## 13. Limiti residui e dati da raccogliere

Restano volutamente non risolti senza acquistare o noleggiare i dispositivi:

- decode Qwen2.5-72B GGUF esatto su ogni Mini PC x86;
- TTFT P50/P95, soprattutto con 4k/8k contesto;
- watt alla presa e rumore dopo 30–60 minuti;
- comportamento reale di 2×48 GB su ogni BIOS;
- differenza Vulkan/ROCm sulla 780M/890M/8060S;
- throughput durante chat + job asincrono;
- prezzo italiano finale con IVA, reso e garanzia;
- stabilità Linux e aggiornamenti firmware dei brand importati.

Questi vuoti non sono un dettaglio editoriale: sono precisamente ciò che impedisce di trasformare la tabella di specifiche in una promessa di 8 tok/s. La procedura corretta è acquistare il candidato con reso, eseguire il protocollo identico e conservare il risultato come criterio di accettazione.
