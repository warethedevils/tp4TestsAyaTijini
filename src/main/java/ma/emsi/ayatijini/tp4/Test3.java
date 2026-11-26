package ma.emsi.ayatijini.tp4;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentParser;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.loader.ClassPathDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.model.google.genai.GoogleGenAiChatModel;
import dev.langchain4j.model.input.Prompt;
import dev.langchain4j.model.input.PromptTemplate;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.query.Query;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import java.time.Duration;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Scanner;
import java.util.logging.ConsoleHandler;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Test 4 : pas de RAG. Un QueryRouter personnalisé demande au LM si la question porte sur le sujet
 * du support de cours ; si ce n'est pas le cas, aucun ContentRetriever n'est retourné et il n'y a pas de RAG
 * (par exemple pour "Bonjour").
 */
public class Test3 {

    private static void configureLogger() {
       // Configure le logger sous-jacent (java.util.logging)
       Logger packageLogger = Logger.getLogger("dev.langchain4j");
       packageLogger.setLevel(Level.FINE); // Ajuster niveau
       // Ajouter un handler pour la console pour faire afficher les logs
       ConsoleHandler handler = new ConsoleHandler();
       handler.setLevel(Level.FINE);
       packageLogger.addHandler(handler);
    }

    public static void main(String[] args) {
        configureLogger();

        String cle = System.getenv("GEMINI_KEY");
        if (cle == null || cle.isBlank()) {
            throw new IllegalStateException("La variable d'environnement GEMINI_KEY n'est pas définie.");
        }

        ChatModel model = GoogleGenAiChatModel.builder()
                .apiKey(cle)
                .modelName(System.getProperty("modele", "gemini-3.5-flash-lite"))
                .temperature(0.3)
                .timeout(Duration.ofSeconds(60))
                .maxRetries(3)
                .logRequestsAndResponses(true)
                .build();

        // ---------- Phase 1 : ingestion du support de cours sur le RAG ----------
        DocumentParser documentParser = new ApacheTikaDocumentParser();
        Document document = ClassPathDocumentLoader.loadDocument("rag.pdf", documentParser);
        DocumentSplitter splitter = DocumentSplitters.recursive(300, 30);
        List<TextSegment> segments = splitter.split(document);

        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();
        List<Embedding> embeddings = embeddingModel.embedAll(segments).content();
        EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();
        embeddingStore.addAll(embeddings, segments);

        // ---------- Phase 2 : RAG ou pas, selon la question ----------
        ContentRetriever contentRetriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore)
                .embeddingModel(embeddingModel)
                .maxResults(2)
                .minScore(0.5)
                .build();

        // Question posée au LM (directement avec le ChatModel, sans assistant) pour décider s'il faut le RAG
        PromptTemplate promptTemplate = PromptTemplate.from(
                "Est-ce que la requête '{{requete}}' porte sur le 'RAG' (Retrieval Augmented Generation) "
                + "ou le 'Fine Tuning' ? Réponds seulement par 'oui', 'non', ou 'peut-être'.");

        class QueryRouterPourEviterRag implements QueryRouter {

          @Override
          public List<ContentRetriever> route(Query query) {
            Prompt question = promptTemplate.apply(Map.of("requete", query.text()));
            String reponse = model.chat(question.text());
            if (reponse.toLowerCase().contains("non")) {
              // Pas de RAG
              return Collections.emptyList();
            } else {
              return Collections.singletonList(contentRetriever);
            }
          }

        }

        QueryRouter queryRouter = new QueryRouterPourEviterRag();

        RetrievalAugmentor retrievalAugmentor = DefaultRetrievalAugmentor.builder()
                .queryRouter(queryRouter)
                .build();

        Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(model)
                .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
                .retrievalAugmentor(retrievalAugmentor)
                .build();

        conversationAvec(assistant);
    }

    private static void conversationAvec(Assistant assistant) {
        try (Scanner scanner = new Scanner(System.in)) {
          while (true) {
            System.out.println("==================================================");
            System.out.println("Posez votre question : ");
            String question = scanner.nextLine();
            if (question.isBlank()) {
              continue;
            }
            System.out.println("==================================================");
            if ("fin".equalsIgnoreCase(question)) {
              break;
            }
            String reponse = assistant.chat(question);
            System.out.println("Assistant : " + reponse);
            System.out.println("==================================================");
          }
        }
    }
}
