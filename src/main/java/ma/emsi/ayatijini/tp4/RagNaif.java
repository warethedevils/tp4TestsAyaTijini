package ma.emsi.ayatijini.tp4;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentParser;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.loader.ClassPathDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.memory.ChatMemory;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.model.google.genai.GoogleGenAiChatModel;
import dev.langchain4j.model.output.Response;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
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
 * Test 1 : RAG "naïf". Les étapes cachées par le RAG facile sont écrites explicitement
 * (parsing du PDF, découpage, embeddings, magasin d'embeddings, récupération).
 */
public class RagNaif {

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
                // -Dmodele=... permet de changer de modèle (quota gratuit épuisé, par exemple)
                .modelName(System.getProperty("modele", "gemini-3.5-flash-lite"))
                .temperature(0.3)
                // sans timeout, une requête restée sans réponse bloque le programme indéfiniment
                .timeout(Duration.ofSeconds(60))
                .maxRetries(3)
                .logRequestsAndResponses(true) // pour voir ce que le RAG ajoute à la question
                .build();

        // ---------- Phase 1 : enregistrement des embeddings ----------

        // Le PDF est dans src/main/resources, donc dans le classpath de l'application
        DocumentParser documentParser = new ApacheTikaDocumentParser();
        Document document = ClassPathDocumentLoader.loadDocument("rag.pdf", documentParser);

        // Découpage en segments de 300 tokens au plus, avec 30 tokens de chevauchement
        DocumentSplitter splitter = DocumentSplitters.recursive(300, 30);
        List<TextSegment> segments = splitter.split(document);

        // Calcul des embeddings des segments (modèle local, sans coût en tokens)
        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();
        Response<List<Embedding>> response = embeddingModel.embedAll(segments);
        List<Embedding> embeddings = response.content();

        // Les embeddings et leurs segments sont enregistrés dans un magasin en mémoire
        EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();
        embeddingStore.addAll(embeddings, segments);

        // ---------- Phase 2 : utilisation des embeddings pour répondre aux questions ----------

        // Les 2 segments les plus pertinents, seulement si leur score est au moins 0,5
        ContentRetriever contentRetriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore)
                .embeddingModel(embeddingModel)
                .maxResults(2)
                .minScore(0.5)
                .build();

        ChatMemory chatMemory = MessageWindowChatMemory.withMaxMessages(10);

        Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(model)
                .chatMemory(chatMemory)
                .contentRetriever(contentRetriever)
                .build();

        conversationAvec(assistant);
    }

    /**
     * Pose plusieurs questions à l'assistant sans recompiler (code fourni dans le TP 2).
     * La conversation s'arrête quand l'utilisateur tape "fin".
     */
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
