from jinja2 import Environment, BaseLoader


DEFAULT_SYSTEM_PROMPT = (
    "You are a research assistant. Answer the question based only on the "
    "provided context. If the context does not contain enough information "
    "to answer, say so."
)

DEFAULT_USER_TEMPLATE = (
    "Context:\n"
    "{% for doc in retrieved_docs %}"
    "- {{ doc }}\n"
    "{% endfor %}\n\n"
    "Question: {{ query }}"
)


class PromptEngine:
    """Build chat messages from retrieved context using Jinja2 templates."""

    def __init__(
        self,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        user_template: str = DEFAULT_USER_TEMPLATE,
    ):
        self.system_prompt = system_prompt
        self.env = Environment(loader=BaseLoader())
        self.user_template = self.env.from_string(user_template)

    def build_messages(
        self,
        query: str,
        retrieved_docs: list[str],
    ) -> list[dict]:
        """Build a chat messages list for the LLM backend."""
        user_content = self.user_template.render(
            query=query,
            retrieved_docs=retrieved_docs,
        )
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]
