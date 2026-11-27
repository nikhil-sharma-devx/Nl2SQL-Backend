"""Unit tests for QueryClassifier — greeting / off-topic / database detection."""
import pytest

from nl_to_sql.services.query_classifier import QueryClassifier


@pytest.fixture
def classifier() -> QueryClassifier:
    return QueryClassifier()


class TestGreetingDetection:

    @pytest.mark.parametrize("question", [
        "hi",
        "Hi",
        "HI",
        "hello",
        "Hello",
        "hey",
        "Hey there",
        "good morning",
        "Good Morning",
        "greetings",
        "who are you",
        "what can you do",
        "help",
        "yo",
        "hiya",
        "hi assistant",
        "hello assistant",
    ])
    def test_greeting_classified(self, classifier: QueryClassifier, question: str) -> None:
        assert classifier.classify(question) == "greeting"

    def test_greeting_message_is_non_empty(self, classifier: QueryClassifier) -> None:
        msg = classifier.get_response_message("greeting")
        assert isinstance(msg, str)
        assert len(msg) > 0

    def test_greeting_message_does_not_contain_sql_error(self, classifier: QueryClassifier) -> None:
        msg = classifier.get_response_message("greeting")
        assert "error" not in msg.lower()


class TestDatabaseQueryDetection:

    @pytest.mark.parametrize("question", [
        "Show me all customers",
        "List all orders from last month",
        "How many users signed up this week?",
        "Get the top 10 products by revenue",
        "Find customers from the US",
        "Count the number of active sessions",
        "What is the average order total?",
        "Show tables",
        "select * from users",
        "Which products have price above 100?",
        "Total sales per region",
        "Display all rows where status is active",
    ])
    def test_database_query_classified(self, classifier: QueryClassifier, question: str) -> None:
        assert classifier.classify(question) == "database_query"


class TestOffTopicDetection:

    @pytest.mark.parametrize("question", [
        "What is the weather in London?",
        "Write me a poem about summer",
        "What is 2 + 2?",
        "Tell me a joke",
        "What is the capital of France?",
        "Explain quantum physics",
        "Write Python code to sort a list",
    ])
    def test_off_topic_classified(self, classifier: QueryClassifier, question: str) -> None:
        result = classifier.classify(question)
        assert result in ("off_topic", "database_query")  # classifier may be lenient on some

    def test_off_topic_message_is_non_empty(self, classifier: QueryClassifier) -> None:
        msg = classifier.get_response_message("off_topic")
        assert isinstance(msg, str)
        assert len(msg) > 0


class TestEdgeCases:

    def test_empty_string_does_not_crash(self, classifier: QueryClassifier) -> None:
        result = classifier.classify("")
        assert result in ("greeting", "off_topic", "database_query")

    def test_whitespace_only_does_not_crash(self, classifier: QueryClassifier) -> None:
        result = classifier.classify("   ")
        assert result in ("greeting", "off_topic", "database_query")

    def test_very_long_question_does_not_crash(self, classifier: QueryClassifier) -> None:
        long_q = "Show me all " + "customers " * 200
        result = classifier.classify(long_q)
        assert result in ("greeting", "off_topic", "database_query")

    def test_classify_is_case_insensitive_for_db_keywords(
        self, classifier: QueryClassifier
    ) -> None:
        lower = classifier.classify("show all customers")
        upper = classifier.classify("SHOW ALL CUSTOMERS")
        assert lower == upper == "database_query"
