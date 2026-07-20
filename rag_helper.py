from http import client
from contextlib import contextmanager

from google.genai.types import GenerateContentConfig


try:
    from opentelemetry import trace
except ImportError:
    trace = None


class _DummySpan:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


class _DummyTracer:
    @contextmanager
    def start_as_current_span(self, name, **kwargs):
        yield _DummySpan()


def _default_tracer():
    if trace is not None:
        return trace.get_tracer(__name__)
    return _DummyTracer()


INSTRUCTIONS = '''
Your task is to answer questions from the course participants
based on the provided context.

Use the context to find relevant information and provide accurate
answers. If the answer is not found in the context,
respond with "I don't know."
'''

PROMPT_TEMPLATE = '''
QUESTION: {question}

CONTEXT:
{context}
'''.strip()


class RAGBase:

    def __init__(
        self,
        index,
        llm_client,
        instructions=INSTRUCTIONS,
        prompt_template=PROMPT_TEMPLATE,
        model='gpt-5.4-mini',
        tracer=None,
    ):
        self.index = index
        self.llm_client = llm_client
        self.instructions = instructions
        self.prompt_template = prompt_template
        self.model = model
        self.tracer = tracer or _default_tracer()

    def search(self, query, num_results=5):
        return self.index.search(query, num_results=num_results)

    def build_context(self, search_results):
        lines = []

        for doc in search_results:
            lines.append(doc['filename'])
            lines.append(doc['content'])
            lines.append('')

        return '\n'.join(lines).strip()

    def build_prompt(self, query, search_results):
        context = self.build_context(search_results)
        return self.prompt_template.format(
            question=query, context=context
        )
    
    def llm(self, prompt):
        response = self.llm_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=GenerateContentConfig(
                response_mime_type="application/json",
                system_instruction=self.instructions,
            ),
        )

        return response

    def rag(self, query):
        search_results = self.search(query)
        prompt = self.build_prompt(query, search_results)
        answer = self.llm(prompt)
        return answer.text


class RAGTraced(RAGBase):

    def search(self, query, num_results=5):
        with self.tracer.start_as_current_span("search"):
            return super().search(query, num_results=num_results)

    def llm(self, prompt):
        with self.tracer.start_as_current_span("llm"):
            return super().llm(prompt)

    def rag(self, query):
        with self.tracer.start_as_current_span("rag"):
            return super().rag(query)
