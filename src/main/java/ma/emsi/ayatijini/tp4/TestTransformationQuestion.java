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
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.query.transformer.CompressingQueryTransformer;
import dev.langchain4j.rag.query.transformer.QueryTransformer;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import java.time.Duration;
import java.util.List;
import java.util.Scanner;
import java.util.logging.ConsoleHandler;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Test 2 : RAG naïf avec transformation de la question.
 * <p>
 * Dans une conversation, une question comme "à quoi c'est comparé ?" ne veut rien dire toute seule :
 * la recherche de similarité faite avec cette seule question ne retrouve pas les bons segments.
 * Le {@link CompressingQueryTransformer} demande au LM de réécrire la question en tenant compte
 * de l'historique de la conversation, avant la recherche des segments.
 */
public class TestTransformationQuestion {

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
                // sans timeout, une requête restée sans réponse bloque le programme indéfiniment
                .timeout(Duration.ofSeconds(60))
                .maxRetries(3)
                .logRequestsAndResponses(true)
                .build();

        // ---------- Phase 1 : ingestion (comme le RAG naïf) ----------
        DocumentParser documentParser = new ApacheTikaDocumentParser();
        Document document = ClassPathDocumentLoader.loadDocument("rag.pdf", documentParser);
        DocumentSplitter splitter = DocumentSplitters.recursive(300, 30);
        List<TextSegment> segments = splitter.split(document);

        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();
        List<Embedding> embeddings = embeddingModel.embedAll(segments).content();
        EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();
        embeddingStore.addAll(embeddings, segments);

        // ---------- Phase 2 : récupération, après transformation de la question ----------
        ContentRetriever contentRetriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore)
                .embeddingModel(embeddingModel)
                .maxResults(2)
                .minScore(0.5)
                .build();

        // Le LM "compresse" la question et l'historique en une question autonome
        QueryTransformer queryTransformer = new CompressingQueryTransformer(model);

        RetrievalAugmentor retrievalAugmentor = DefaultRetrievalAugmentor.builder()
                .queryTransformer(queryTransformer)
                .contentRetriever(contentRetriever)
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
