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
import dev.langchain4j.rag.query.router.LanguageModelQueryRouter;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import java.time.Duration;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;
import java.util.logging.ConsoleHandler;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Test 3 : routage. Deux documents sur des thèmes différents sont enregistrés dans deux magasins
 * d'embeddings ; le LM choisit, pour chaque question, dans quelle source chercher.
 */
public class TestRoutage {

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

        // Le même modèle sert à répondre aux questions et à faire le routage
        ChatModel model = GoogleGenAiChatModel.builder()
                .apiKey(cle)
                .modelName(System.getProperty("modele", "gemini-3.5-flash-lite"))
                .temperature(0.3)
                .timeout(Duration.ofSeconds(60))
                .maxRetries(3)
                .logRequestsAndResponses(true)
                .build();

        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();

        // ---------- Phase 1 : un magasin d'embeddings par document ----------
        EmbeddingStore<TextSegment> storeIa = ingerer("rag.pdf", embeddingModel);
        EmbeddingStore<TextSegment> storeCuisine = ingerer("cuisine-marocaine.pdf", embeddingModel);

        // ---------- Phase 2 : un ContentRetriever par source, et le routage ----------
        ContentRetriever retrieverIa = creerContentRetriever(storeIa, embeddingModel);
        ContentRetriever retrieverCuisine = creerContentRetriever(storeCuisine, embeddingModel);

        // Descriptions en langage naturel, lues par le LM pour choisir la (ou les) source(s)
        Map<ContentRetriever, String> descriptions = new HashMap<>();
        descriptions.put(retrieverIa,
                "Support de cours sur l'intelligence artificielle : fine-tuning des modèles de langage, "
                + "RAG (Retrieval-Augmented Generation), embeddings et LangChain4j");
        descriptions.put(retrieverCuisine,
                "Document sur la cuisine marocaine : plats traditionnels (tajine, couscous, harira, pastilla), "
                + "thé à la menthe, épices, pâtisseries et traditions des repas");

        QueryRouter queryRouter = new LanguageModelQueryRouter(model, descriptions);

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

    /**
     * Phase 1 du RAG pour un fichier de ressources : parsing, découpage en segments,
     * calcul des embeddings et enregistrement dans un nouveau magasin en mémoire.
     */
    private static EmbeddingStore<TextSegment> ingerer(String nomFichier, EmbeddingModel embeddingModel) {
        DocumentParser documentParser = new ApacheTikaDocumentParser();
        Document document = ClassPathDocumentLoader.loadDocument(nomFichier, documentParser);

        DocumentSplitter splitter = DocumentSplitters.recursive(300, 30);
        List<TextSegment> segments = splitter.split(document);

        List<Embedding> embeddings = embeddingModel.embedAll(segments).content();
        EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();
        embeddingStore.addAll(embeddings, segments);
        return embeddingStore;
    }

    private static ContentRetriever creerContentRetriever(EmbeddingStore<TextSegment> embeddingStore,
                                                          EmbeddingModel embeddingModel) {
        return EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore)
                .embeddingModel(embeddingModel)
                .maxResults(2)
                .minScore(0.5)
                .build();
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
