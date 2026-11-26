package ma.emsi.ayatijini.tp4;

/**
 * Service IA utilisé par les tests du TP : LangChain4j (AiServices) en fournit l'implémentation.
 */
public interface Assistant {

    /**
     * Envoie un message de l'utilisateur au LM et retourne sa réponse.
     */
    String chat(String userMessage);
}
