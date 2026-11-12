# Avaliação manual da demo (LGPD)

Teste manual feito por mim na demo pública, em novembro de 2026, com o PDF de exemplo da LGPD (edição do Senado Federal atualizada até junho de 2024). Foram 12 perguntas: 9 diretas ou amplas e 3 fora do documento. Em cada uma conferi a resposta no PDF e verifiquei em "Ver trechos usados" se o trecho certo foi recuperado.

Configuração: chunks de 1000 caracteres com 200 de sobreposição, 5 trechos recuperados (k=5), modelo gemini-3.1-flash-lite.

## Resultado

| # | Pergunta | Resultado | Páginas citadas | Trecho certo recuperado | Modo | Observação |
|---|---|---|---|---|---|---|
| 1 | Qual é o valor máximo da multa simples por infração? | Correta | 35 | Sim | literal | 2% do faturamento, limitada a R$ 50 milhões |
| 2 | O que é dado pessoal sensível? | Correta | 10 | Sim | literal | |
| 3 | O que é o encarregado pelo tratamento de dados pessoais? | Correta | 11 | Sim | literal | Também recuperou o índice (p. 5), irrelevante |
| 4 | Em quanto tempo o controlador deve responder ao titular que pede a confirmação da existência de tratamento? | Correta | 21, 23 | Sim | literal | O art. 19 começa na p. 22, e a citação não a incluiu |
| 5 | Como deve ser o consentimento para tratar dados de crianças? | Correta | 19 | Sim | literal | |
| 6 | Em quais casos a LGPD não se aplica? | Correta | 9, 10 | Sim | literal | Lista completa do art. 4 |
| 7 | Quais são as sanções administrativas que a ANPD pode aplicar? | Parcial | 35 | Parcial | literal | Listou só advertência e multa simples. A p. 36 (demais sanções) não foi recuperada, e a resposta parece completa |
| 8 | Quais são as bases legais para o tratamento de dados pessoais? | Correta | 13, 14 | Sim | parafraseado | O modo literal veio vazio e o prompt reserva resolveu. Listou as 10 hipóteses do art. 7 |
| 9 | Qual é a diferença entre controlador e operador? | Correta | 11, 30 | Sim | literal | |
| 10 | Qual é a capital da França? | Recusou (correto) | | | literal | Respondeu "Não sei com base no documento." |
| 11 | Em quantos dias úteis o incidente de segurança deve ser comunicado à ANPD? | Correta | 33 | Sim | literal | Disse que a lei fala em "prazo razoável" (art. 48), sem inventar número |
| 12 | Quais são as multas do GDPR europeu? | Recusou (correto) | | | literal | Respondeu "Não sei com base no documento." |

Total: 9 respostas corretas, 2 recusas corretas, 1 parcial e nenhuma resposta inventada.

## O que o teste mostrou

- **Perguntas amplas dependem do k.** Na pergunta 7, as sanções estão em vários trechos e um deles ficou de fora. O app não avisa que a lista pode estar incompleta.
- **A citação de página pode ser imprecisa.** Na pergunta 4, a página 22 ficou fora da citação.
- **Perguntas fora do documento recuperam trechos irrelevantes** (capa e folha de rosto na pergunta 10), porque não há limiar de similaridade. O prompt é que garante a recusa.
- **O prompt reserva foi necessário em 1 de 12 perguntas.**

## Limitações desta avaliação

Amostra pequena, um único PDF, e quem avaliou foi o próprio autor. Serve como verificação de sanidade, não como benchmark.
