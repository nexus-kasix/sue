import asyncio
from crawl4ai import AsyncWebCrawler
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from typing import Dict, Any, Optional
import logging
from googlesearch import search
from config import config
from utils import web_cache, process_webpage_async, search_web

# Настройка логгера
logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)

class SueAgent:
    """
    Sue AI Assistant implementation.
    """

    def __init__(self):
        """Initialize the agent."""
        self._setup_logo()
        self._setup_memory()
        self._setup_context_storage()
        self._setup_llm()

    def _setup_logo(self):
        """Setup ASCII logo."""
        self.ascii_logo = """
#######   ###     ###   #######   ###         ###   #######   ######     
###       ###     ###   ###       ###   ###   ###   ###       ###  ###     
#######   ###     ###   #######   ###  ### ## ###   ######    ######    
    ###   ###     ###   ###       ### ###  ### ###   ###       ###  ###    
#######    #######     #######    #####      #####   #######   ######     

   ####      ######    #######   ######    ###   #######
  ### ###   ###        ###       ###  ###  ###     ###
 #########  ######     ######    ###  ###  ###     ###
###     ### ###        ###       ###  ###  ###     ###
###     ### #######    #######   ######    ###     ###
        """

    def _setup_llm(self):
        """Setup language model and chain."""
        # Validate configuration
        issues = config.validate()
        if issues:
            raise ValueError(f"Configuration issues found: {issues}")

        # Initialize model
        self.llm = ChatMistralAI(
            api_key=config.MISTRAL_API_KEY,
            model=config.MODEL_NAME,
            temperature=config.TEMPERATURE,
            max_tokens=config.MAX_TOKENS
        )

        # Create prompt template
        system_prompt = """You are Sue, an AI assistant that can search the internet, solve examples, etc. 
Always respond in Russian language. Be polite and cultured, without any slang. 
But if the user asks you to be youthful, be more casual in your responses. 
Remember the context of the conversation and previous interactions."""

        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_prompt),
            MessagesPlaceholder(variable_name="history"),
            HumanMessagePromptTemplate.from_template("{input}")
        ])

        # Setup chain
        self.output_parser = StrOutputParser()
        self.chain = self.prompt | self.llm | self.output_parser

    def _setup_memory(self):
        """Setup conversation memory."""
        self.chat_history = ChatMessageHistory()

    def _setup_context_storage(self):
        """Setup context storage."""
        self.context_pages: Dict[str, Dict[str, Any]] = {}
        self.current_context: Optional[str] = None
        self.last_url: Optional[str] = None

    def _get_chat_history(self) -> Dict[str, Any]:
        """Get formatted chat history."""
        return {"history": self.chat_history.messages}

    def _add_to_history(self, human_message: str, ai_message: str) -> None:
        """Add messages to chat history."""
        self.chat_history.add_message(HumanMessage(content=human_message))
        self.chat_history.add_message(AIMessage(content=ai_message))

    def chat(self, user_input: str) -> str:
        """
        Process chat input with error handling.
        """
        try:
            # Default chat behavior
            chat_prompt = f"""Ответь на следующее сообщение пользователя, учитывая предыдущий контекст разговора.
Сообщение пользователя: {user_input}"""

            response = self.chain.invoke({
                "input": chat_prompt,
                **self._get_chat_history()
            })

            self._add_to_history(user_input, response)
            return response

        except Exception as e:
            logger.error(f"Chat error: {str(e)}")
            return f"Произошла ошибка: {str(e)}"

    async def handle_md_web_command(self, url: str) -> str:
        """
        Обрабатывает команду /md_web, извлекает markdown и добавляет его в контекст.
        """
        try:
            markdown_content = await process_webpage_async(url)

            if markdown_content.startswith("Ошибка"):
                return markdown_content

            self.context_pages[url] = {
                'content': markdown_content,
                'timestamp': asyncio.get_event_loop().time()
            }
            self.current_context = url
            self.last_url = url
            logger.info(f"Added {url} to context (markdown)")
            return f"Контент страницы {url} добавлен в контекст в формате markdown."

        except Exception as e:
            logger.error(f"Error processing webpage: {str(e)}")
            return f"Произошла ошибка: {str(e)}"

    async def search_and_respond(self, query: str) -> str:
        """Search the web and respond to query"""
        try:
            results = await search_web(query, num_results=3)  # Максимум 3 результата
            if not results:
                return "Извините, не удалось найти информацию по вашему запросу."

            formatted_results = ""
            for i, result in enumerate(results, 1):
                formatted_results += f"""
Результат {i}:
Заголовок: {result['title']}
URL: {result['link']}
Описание: {result['snippet']}
"""

            search_prompt = f"""На основе следующих результатов поиска ответь на вопрос пользователя.
Дай полный и информативный ответ, используя найденную информацию.

Вопрос: {query}

Результаты поиска:
{formatted_results}

Пожалуйста, дай структурированный ответ на русском языке."""

            response = self.chain.invoke({
                "input": search_prompt,
                **self._get_chat_history()
            })
            
            self._add_to_history(query, response)
            return response

        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            return f"Произошла ошибка при поиске: {str(e)}"

    async def process_webpage(self, url: str, prompt: str, store_context: bool = True) -> str:
        """Process webpage content with given prompt"""
        try:
            markdown_content = await process_webpage_async(url)

            if markdown_content.startswith("Ошибка"):
                return markdown_content

            # Store in context if requested
            if store_context:
                self.context_pages[url] = {
                    'content': markdown_content,
                    'timestamp': asyncio.get_event_loop().time()
                }
                self.current_context = url
                self.last_url = url
                logger.info(f"Added {url} to context (markdown)")

            context_prompt = f"""Проанализируй следующий текст и ответь на запрос, учитывая историю разговора.
Запрос: {prompt}

Содержимое страницы:
{markdown_content}

Пожалуйста, отвечай на основе предоставленного текста и истории разговора."""

            response = self.chain.invoke({
                "input": context_prompt,
                **self._get_chat_history()
            })

            self._add_to_history(prompt, response)
            return response

        except Exception as e:
            logger.error(f"Error processing webpage: {str(e)}")
            return f"Ошибка при обработке страницы: {str(e)}"

    def chat_with_context(self, user_input: str) -> str:
        """Chat with current context"""
        try:
            if not self.current_context or not self.context_pages:
                return self.chat(user_input)

            # Check for general chat phrases
            general_chat_phrases = [
                "как дела", "привет", "здравствуй", "пока", "до свидания",
                "спасибо", "благодарю", "доброе утро", "добрый день", "добрый вечер"
            ]

            if any(phrase in user_input.lower() for phrase in general_chat_phrases):
                return self.chat(user_input)

            # Format context
            context_content = []
            for url, data in self.context_pages.items():
                context_content.append(f"=== Контекст из {url} ===\n{data['content']}")

            full_context = "\n\n".join(context_content)

            context_prompt = f"""Используй следующий контекст и историю разговора для ответа на вопрос.
Вопрос: {user_input}

Контекст:
{full_context}

Если вопрос не связан с контекстом или это общий вопрос, используй историю разговора."""

            response = self.chain.invoke({
                "input": context_prompt,
                **self._get_chat_history()
            })

            self._add_to_history(user_input, response)
            return response

        except Exception as e:
            logger.error(f"Chat error: {str(e)}")
            return f"Произошла ошибка: {str(e)}"

    def display_welcome(self) -> str:
        """Display welcome message with ASCII logo"""
        return f"""
{self.ascii_logo}

Доступные команды:
/q - Выход из чата
/web [URL] [ЗАПРОС] - Анализ содержимого веб-страницы
/c_web [URL1,URL2,...] - Загрузить страницы в контекст
/s_web [ЗАПРОС] - Поиск информации в интернете
/md_web [URL] - Извлечь контент в markdown и добавить в контекст
После загрузки контекста можно общаться напрямую по содержимому страниц
Чем могу помочь?"""


async def main():
    """Main entry point"""
    try:
        agent = SueAgent()
        print(agent.display_welcome())

        while True:
            try:
                user_input = input("\nВы: ").strip()

                if user_input.lower() == '/q':
                    print("До свидания!")
                    break

                elif user_input.startswith('/md_web '):
                    url = user_input[8:].strip()
                    if not url:
                        print("Использование: /md_web [URL]")
                        continue
                    response = await agent.handle_md_web_command(url)
                    print(f"\nSue: {response}")

                elif user_input.startswith('/web '):
                    parts = user_input[5:].split(' ', 1)
                    if len(parts) != 2:
                        print("Использование: /web [URL] [ЗАПРОС]")
                        continue
                    url, query = parts
                    response = await agent.process_webpage(url.strip(), query.strip())
                    print(f"\nSue: {response}")

                elif user_input.startswith('/c_web '):
                    urls = user_input[7:].split(',')
                    if not urls:
                        print("Использование: /c_web [URL1,URL2,...]")
                        continue
                    for url in urls:
                        url = url.strip()
                        if url:
                            response = await agent.process_webpage(url, "Загрузка страницы в контекст", store_context=True)
                            print(f"Загружена страница: {url}")
                    print("\nSue: Страницы загружены в контекст. Можете задавать вопросы.")

                elif user_input.startswith('/s_web '):
                    query = user_input[7:].strip()
                    if not query:
                        print("Использование: /s_web [ЗАПРОС]")
                        continue
                    print("\nSue: Ищу информацию...")
                    response = await agent.search_and_respond(query)
                    print(f"\nSue: {response}")

                else:
                    response = agent.chat_with_context(user_input)
                    print(f"\nSue: {response}")

            except (EOFError, KeyboardInterrupt):
                print("\nДо свидания!")
                break

    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        print(f"\nПроизошла ошибка: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())
