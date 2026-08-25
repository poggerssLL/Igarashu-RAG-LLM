# IA local de estudos com RAG

Este projeto oferece uma IA local para apoiar a graduação em Engenharia de Controle e Automação. Ele indexa PDFs, recupera os trechos mais relevantes e responde às perguntas indicando o arquivo e a página usados como fonte.

Todo o processamento será feito no computador: o Ollama executa os modelos, o ChromaDB armazena os vetores localmente e nenhum serviço de IA em nuvem é necessário.

## Requisitos

- Windows com PowerShell;
- Python 3.11 ou superior (versão mínima declarada pelo projeto);
- Ollama em execução;
- modelo `qwen2.5:3b` e um modelo de embeddings compatível instalado:
  `nomic-embed-text` (padrão) ou `embeddinggemma`.

O ambiente funcional atual foi usado e validado com **Python 3.12.13**. O
Python 3.11 permanece como versão mínima declarada, mas a reprodução exata
registrada neste repositório corresponde ao Python 3.12.13 no Windows.

Para conferir os modelos:

```powershell
ollama list
```

## Ativar o ambiente virtual

Na raiz do projeto, execute:

```powershell
.\.venv\Scripts\Activate.ps1
```

Se o PowerShell bloquear o script apenas nesta sessão, use:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Para sair do ambiente virtual:

```powershell
deactivate
```

## Instalar as dependências

O `requirements.txt` contém somente as dependências diretas do projeto, com
as versões do ambiente funcional atual. Para a instalação normal, com o
ambiente ativado, execute:

```powershell
python -m pip install -r requirements.txt
```

Para reproduzir exatamente o conjunto completo validado no Windows com Python
3.12.13, incluindo as dependências transitivas, use:

```powershell
python -m pip install -r requirements-lock.txt
```

O `requirements-lock.txt` representa uma fotografia completa da `.venv`
validada. Os modelos do Ollama não são dependências Python e não fazem parte
desses arquivos. Instalar qualquer um dos arquivos de requisitos não baixa
automaticamente `qwen2.5:3b`, `nomic-embed-text`, `embeddinggemma` ou outros
modelos do Ollama.

## Testar o Ollama

Teste o modelo de conversa:

```powershell
ollama run qwen2.5:3b "Responda em português: o que é um sistema de controle?"
```

Teste o modelo de embeddings pela biblioteca Python:

```powershell
python -c "import ollama; r = ollama.embed(model='nomic-embed-text', input='Teste de embeddings'); print(len(r['embeddings'][0]))"
```

O segundo comando deve imprimir `768`, que é o número de dimensões do vetor gerado pelo `nomic-embed-text`.

## Estrutura

- `Documentos/`: PDFs e outros materiais pessoais de estudo;
- `Dados/`: índices e dados gerados pelo RAG;
- `src/`: código-fonte;
- `avaliacao/`: testes e avaliações do sistema.

Os conteúdos de `Documentos/` e `Dados/` são ignorados pelo Git.

## Ingerir os PDFs

Coloque os arquivos `.pdf` em `Documentos/` (subpastas também são aceitas), mantenha o Ollama em execução e rode, na raiz do projeto:

```powershell
.\.venv\Scripts\python.exe -m src.ingest
```

O comando extrai o texto página por página, divide-o em trechos sobrepostos, gera
embeddings com o modelo configurado (por padrão, `nomic-embed-text`) e faz `upsert` no
ChromaDB persistente em `Dados/chroma`. Ao final, ele informa quantos PDFs, páginas e
trechos foram processados.

PDFs digitalizados apenas como imagem não possuem texto extraível. Nesse caso, aplique OCR ao documento antes de executar a ingestão.

## Consultar o material indexado

Faça uma pergunta na raiz do projeto:

```powershell
.\.venv\Scripts\python.exe -m src.chat "O que é o teorema da amostragem?"
```

Por padrão, a consulta recupera quatro trechos. Para alterar essa quantidade:

```powershell
.\.venv\Scripts\python.exe -m src.chat "Como um sinal pode ser reconstruído?" --top-k 6
```

Para inspecionar os textos entregues ao modelo antes da resposta:

```powershell
.\.venv\Scripts\python.exe -m src.chat "Qual é a frequência mínima de amostragem?" --mostrar-contexto
```

A resposta é gerada localmente pelo `qwen2.5:3b` e deve terminar com uma seção `Fontes`. Se os trechos recuperados não contiverem informação suficiente, o modelo é instruído a declarar que não encontrou a resposta no material indexado.

### Modo fundamentado em um único PDF

O modo padrão de consulta é **Fundamentado**. Ele não faz síntese entre vários PDFs:

1. recupera até 20 candidatos e escolhe o documento com as evidências mais fortes;
2. repete a recuperação filtrando simultaneamente matéria e campo `arquivo`;
3. seleciona de quatro a seis trechos fortes, remove duplicatas e considera continuações
   em páginas vizinhas;
4. pede ao Qwen para organizar fatos, definições, fórmulas, condições e limitações;
5. redige a resposta no nível de detalhe solicitado;
6. audita as afirmações e publica somente as classificadas como sustentadas;
7. adiciona programaticamente apenas citações de páginas recuperadas.

Na interface, o seletor **Fonte** oferece `Automático` e os PDFs indexados na matéria.
O motivo da escolha automática é mostrado acima da resposta. O seletor **Modo de
resposta** permite voltar ao fluxo anterior usando `Compatibilidade`.

Exemplo automático e curto:

```powershell
.\.venv\Scripts\python.exe -m src.chat "O que caracteriza os sinais periódicos?" --modo fundamentado --disciplina "Sinais e Sistemas" --nivel-detalhe "Curto" --paginas-vizinhas
```

Para escolher um único PDF explicitamente:

```powershell
.\.venv\Scripts\python.exe -m src.chat "Qual é o período fundamental?" --modo fundamentado --disciplina "Sinais e Sistemas" --arquivo "Sinais e Sistemas/Signals_and_Systems_2nd_Edition_by_Oppen.pdf" --nivel-detalhe "Explicado" --mostrar-evidencias
```

Os níveis disponíveis são `Curto`, `Explicado` e `Passo a passo`. Fórmulas e exemplos
só são incluídos quando aparecem nas evidências. Afirmações classificadas como
parcialmente sustentadas ou não sustentadas não são publicadas; continuam disponíveis
na auditoria interna. Quando falta evidência, o sistema informa o que está faltando.

Para executar explicitamente o comportamento anterior:

```powershell
.\.venv\Scripts\python.exe -m src.chat "Sua pergunta" --modo compatibilidade
```

As fontes são verificadas e formatadas automaticamente como
`[arquivo, página do PDF X]`, sem referências duplicadas. Para impedir a geração
quando a busca tiver baixa relevância, informe um limiar entre 0 e 1:

```powershell
.\.venv\Scripts\python.exe -m src.chat "Sua pergunta" --min-relevancia 0.55
```

O valor é calculado a partir da distância L2 do ChromaDB por `1 / (1 + distância)`: quanto mais próximo de 1, melhor. O padrão `0` desativa o bloqueio. Com o material atual, um ponto inicial razoável para experimentar é `0.55`; ajuste-o observando `--mostrar-contexto`, pois um limiar alto pode rejeitar perguntas válidas.

## Avaliar a recuperação

Execute os casos definidos em `avaliacao/casos_rag.json`:

```powershell
.\.venv\Scripts\python.exe -m src.evaluate
```

O avaliador usa o modelo registrado no manifesto do índice e a busca do ChromaDB;
ele não chama o modelo de conversa. Para cada pergunta, mostra as páginas esperadas,
as páginas e os trechos recuperados e os termos encontrados no contexto retornado.

## Recuperação híbrida e multilíngue

A consulta reúne no mínimo 20 candidatos vetoriais e lexicais, combina os rankings e
só então escolhe os trechos enviados ao modelo. A busca por palavras-chave inclui uma
expansão bilíngue pequena para termos técnicos frequentes; a ponte semântica geral
continua sendo responsabilidade do modelo de embeddings.

Exemplo com mais candidatos, páginas vizinhas e resposta em português:

```powershell
.\.venv\Scripts\python.exe -m src.chat "What characterizes periodic signals?" --candidatos 40 --paginas-vizinhas --idioma "Português" --mostrar-contexto
```

Opções importantes:

- `--candidatos N`: quantidade inicial, nunca inferior a 20;
- `--sem-busca-hibrida`: usa somente o ranking vetorial;
- `--paginas-vizinhas`: acrescenta trechos das páginas anterior e seguinte;
- `--sem-diversificacao`: desativa a diversidade condicional por arquivo;
- `--idioma "Português"` ou `--idioma "English"`: fixa o idioma da resposta;
- `--modelo-embeddings NOME`: seleciona o modelo, desde que seja o mesmo do índice.

A diversificação só promove outro arquivo quando sua pontuação é semelhante à do
próximo candidato. Assim, uma fonte menos relevante não é incluída apenas para variar.

### Manifesto e troca do modelo de embeddings

Cada reindexação grava `Dados/manifesto_indice.json` com modelo, dimensão, tamanho e
sobreposição dos trechos, data UTC, coleção e quantidade de vetores. Consultas e novas
ingestões são bloqueadas quando o modelo configurado não coincide com o manifesto.
Isso impede misturar vetores de modelos ou dimensões diferentes.

Para instalar localmente o modelo multilíngue opcional:

```powershell
ollama pull embeddinggemma
```

A troca exige obrigatoriamente recriação completa e confirmação explícita:

```powershell
.\.venv\Scripts\python.exe -m src.ingest --reindexar-tudo --confirmar --modelo-embeddings embeddinggemma
```

Para voltar ao modelo padrão, a mesma proteção se aplica:

```powershell
.\.venv\Scripts\python.exe -m src.ingest --reindexar-tudo --confirmar --modelo-embeddings nomic-embed-text
```

Esses comandos recriam somente o índice vetorial; os PDFs nunca são removidos.

A **taxa de acerto da recuperação** é a proporção de casos em que pelo menos uma página esperada aparece entre os quatro primeiros resultados. Uma falha pode indicar pergunta ambígua, trecho mal dividido, texto mal extraído do PDF ou baixa separação semântica dos embeddings. A contagem de termos é uma verificação complementar: mesmo com a página correta, o trecho recuperado pode não conter todos os detalhes esperados.

### Avaliar a geração e a sustentação factual

Os casos em `avaliacao/casos_geracao.json` cobrem resposta direta, reformulação,
pergunta em português sobre PDF inglês, fórmula, ausência de resposta, indução a inventar
e continuação em página vizinha.

```powershell
.\.venv\Scripts\python.exe -m src.evaluate --geracao
```

Para medir também o fluxo anterior e atualizar
`avaliacao/linha_base_geracao.json`:

```powershell
.\.venv\Scripts\python.exe -m src.evaluate --geracao --comparar-compatibilidade
```

As métricas são calculadas separadamente para página correta, conceitos, validade das
citações, idioma, recusa correta e afirmações não sustentadas. Essa avaliação chama o
Qwen várias vezes e pode demorar alguns minutos; ela é separada da avaliação rápida de
recuperação na interface.

## Interface local

Com o Ollama em execução, abra a interface a partir da raiz do projeto:

```powershell
.\.venv\Scripts\streamlit.exe run src/app.py
```

O navegador normalmente abre automaticamente. Se isso não acontecer, acesse `http://localhost:8501`.

Na área **Documentos**, selecione os PDFs e clique em **Salvar PDFs selecionados**. Depois clique em **Indexar documentos** para extrair o texto e atualizar o ChromaDB. Arquivos existentes só são substituídos quando a confirmação explícita estiver marcada; a interface não oferece exclusão de documentos ou do banco.

Use **Conversar** para escolher disciplina, PDF, modo da resposta, idioma, nível de detalhe, quantidade de candidatos,
busca híbrida, páginas vizinhas, diversificação, relevância mínima e exibição do
contexto ou das evidências organizadas. O idioma padrão é **Português** e o histórico
existe somente durante a sessão atual. Em **Avaliação**, há abas independentes para
recuperação e geração fundamentada.

Em **Documentos**, o seletor de modelo mostra `nomic-embed-text` e `embeddinggemma`.
Escolher um modelo diferente do índice só tem efeito ao marcar a confirmação e usar
**Reindexar toda a biblioteca**; a indexação incremental será bloqueada.

### Organizar por matérias

A área **Matérias** mantém um registro local em `Dados/disciplinas.json`. Cada matéria possui nome, descrição opcional e data de criação, além de uma pasta própria:

```text
Documentos/
  Sinais e Sistemas/
  Controle Linear/
  Eletrônica/
  Automação Industrial/
```

É possível criar e editar matérias, renomear a matéria junto com sua pasta, mover PDFs e excluir matérias vazias. Renomeações, movimentações e exclusões exigem confirmação. Uma matéria com arquivos não pode ser excluída, e nenhum PDF é apagado automaticamente.

PDFs diretamente em `Documentos/` pertencem a **Sem disciplina**. Em subpastas, a primeira pasta após `Documentos/` define a matéria. Por exemplo, `Documentos/Sinais e Sistemas/aula.pdf` recebe a disciplina `Sinais e Sistemas` nos metadados do ChromaDB.

Na área **Documentos**, escolha uma matéria antes de enviar PDFs. Depois de criar, renomear ou mover arquivos entre matérias, marque a confirmação e clique em **Reindexar toda a biblioteca**. Essa ação recria somente o índice vetorial em `Dados/chroma`; os PDFs nunca são apagados.

Também é possível reindexar pela CLI, com confirmação explícita:

```powershell
.\.venv\Scripts\python.exe -m src.ingest --reindexar-tudo --confirmar
```

### Filtrar a busca por matéria

Na barra lateral de **Conversar**, use o seletor **Disciplina**. O padrão **Todas as disciplinas** pesquisa a biblioteca inteira; **Sem disciplina** considera PDFs soltos na raiz; as demais opções consideram somente matérias com trechos indexados.

O mesmo filtro está disponível nas CLIs:

```powershell
.\.venv\Scripts\python.exe -m src.chat "Explique o teorema da amostragem" --disciplina "Sinais e Sistemas"
.\.venv\Scripts\python.exe -m src.evaluate --disciplina "Sinais e Sistemas"
```

Para encerrar o servidor, volte ao PowerShell em que ele está sendo executado e pressione `Ctrl+C`. A interface é exclusivamente local e não é publicada na internet.
