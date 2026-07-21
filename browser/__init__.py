"""
Browser Automation System.

Features:
- Web scraping and content extraction
- Form filling and submission
- Screenshot capture
- JavaScript execution
- Element interaction (click, type, scroll)
- Session management
- Anti-detection measures
- Proxy support
- Headless and headed modes

Supports multiple backends:
- Playwright (recommended)
- Selenium
- requests + BeautifulSoup (basic)
"""

import argparse
import os
import sys
import json
import time
import asyncio
import hashlib
from dataclasses import dataclass, field
from typing import Any, Optional, Callable
from pathlib import Path
from enum import Enum
from urllib.parse import urljoin, urlparse


class BrowserBackend(Enum):
    """Browser automation backends."""
    PLAYWRIGHT = "playwright"
    SELENIUM = "selenium"
    REQUESTS = "requests"


class ElementType(Enum):
    """HTML element types for interaction."""
    BUTTON = "button"
    LINK = "link"
    INPUT = "input"
    TEXTAREA = "textarea"
    SELECT = "select"
    CHECKBOX = "checkbox"
    RADIO = "radio"
    IMAGE = "image"
    TEXT = "text"
    ANY = "any"


@dataclass
class BrowserConfig:
    """Browser configuration."""
    backend: BrowserBackend = BrowserBackend.PLAYWRIGHT
    headless: bool = True
    browser_type: str = "chromium"  # chromium, firefox, webkit
    timeout: int = 30000  # ms
    viewport_width: int = 1280
    viewport_height: int = 720
    user_agent: str = None
    proxy: str = None
    downloads_dir: str = None
    screenshots_dir: str = "./screenshots"
    ignore_https_errors: bool = False
    java_script_enabled: bool = True
    locale: str = "en-US"
    timezone_id: str = "America/New_York"
    extra_args: list[str] = field(default_factory=list)


@dataclass
class ElementInfo:
    """Information about a found element."""
    tag: str
    text: str
    attributes: dict
    selector: str
    index: int
    visible: bool
    enabled: bool
    bbox: Optional[dict] = None


@dataclass
class PageResult:
    """Result of a page operation."""
    success: bool
    url: str
    title: str
    content: str
    elements: list[ElementInfo]
    screenshot: Optional[str] = None
    error: Optional[str] = None
    duration_ms: float = 0


@dataclass
class ScrapedContent:
    """Scraped web content."""
    url: str
    title: str
    text: str
    html: str
    links: list[dict]
    images: list[dict]
    metadata: dict
    timestamp: float = field(default_factory=time.time)


class BrowserAutomation:
    """
    Browser automation system for web interaction.

    Supports Playwright, Selenium, and requests backends.
    """

    def __init__(self, config: BrowserConfig = None):
        self.config = config or BrowserConfig()
        self.browser = None
        self.page = None
        self.context = None
        self.session = None
        self._history: list[dict] = []
        self._cookies: dict = {}

    # === Lifecycle ===

    async def start(self):
        """Start the browser."""
        if self.config.backend == BrowserBackend.PLAYWRIGHT:
            await self._start_playwright()
        elif self.config.backend == BrowserBackend.SELENIUM:
            await self._start_selenium()
        else:
            await self._start_requests()

    async def stop(self):
        """Stop the browser."""
        if self.browser:
            await self.browser.close()
        if self.context:
            await self.context.close()
        if self.session:
            await self.session.close()

    async def _start_playwright(self):
        """Start Playwright browser."""
        try:
            from playwright.async_api import async_playwright
            self._playwright = await async_playwright().start()

            browser_type = getattr(self._playwright, self.config.browser_type)
            launch_args = [
                "--disable-blink-features=AutomationControlled",
                "--disable-infobars",
                "--no-sandbox",
            ] + self.config.extra_args

            launch_kwargs = {
                "headless": self.config.headless,
                "args": launch_args,
            }

            if self.config.proxy:
                launch_kwargs["proxy"] = {"server": self.config.proxy}

            self.browser = await browser_type.launch(**launch_kwargs)

            context_kwargs = {
                "viewport": {
                    "width": self.config.viewport_width,
                    "height": self.config.viewport_height
                },
                "locale": self.config.locale,
                "timezone_id": self.config.timezone_id,
                "ignore_https_errors": self.config.ignore_https_errors,
            }

            if self.config.user_agent:
                context_kwargs["user_agent"] = self.config.user_agent

            self.context = await self.browser.new_context(**context_kwargs)
            self.page = await self.context.new_page()

        except ImportError:
            raise ImportError("Playwright not installed. Run: pip install playwright && playwright install")

    async def _start_selenium(self):
        """Start Selenium browser."""
        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options
            from selenium.webdriver.chrome.service import Service

            options = Options()
            if self.config.headless:
                options.add_argument("--headless")

            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument(f"--window-size={self.config.viewport_width},{self.config.viewport_height}")

            if self.config.proxy:
                options.add_argument(f"--proxy-server={self.config.proxy}")

            self._selenium_driver = webdriver.Chrome(options=options)
            self.page = self._selenium_driver

        except ImportError:
            raise ImportError("Selenium not installed. Run: pip install selenium")

    async def _start_requests(self):
        """Start requests session."""
        try:
            import aiohttp
            self.session = aiohttp.ClientSession()
        except ImportError:
            raise ImportError("aiohttp not installed. Run: pip install aiohttp")

    # === Navigation ===

    async def navigate(self, url: str, wait_until: str = "load") -> PageResult:
        """Navigate to a URL."""
        start_time = time.time()

        try:
            if self.config.backend == BrowserBackend.PLAYWRIGHT:
                response = await self.page.goto(url, wait_until=wait_until)
                title = await self.page.title()
                content = await self.page.content()

            elif self.config.backend == BrowserBackend.SELENIUM:
                self.page.get(url)
                title = self.page.title
                content = self.page.page_source

            else:
                async with self.session.get(url) as resp:
                    content = await resp.text()
                    title = ""
                    url = str(resp.url)

            elements = await self._find_elements()

            result = PageResult(
                success=True,
                url=url,
                title=title,
                content=content,
                elements=elements,
                duration_ms=(time.time() - start_time) * 1000
            )

            self._history.append({
                "action": "navigate",
                "url": url,
                "timestamp": time.time()
            })

            return result

        except Exception as e:
            return PageResult(
                success=False,
                url=url,
                title="",
                content="",
                elements=[],
                error=str(e),
                duration_ms=(time.time() - start_time) * 1000
            )

    async def go_back(self):
        """Navigate back."""
        if self.config.backend == BrowserBackend.PLAYWRIGHT:
            await self.page.go_back()
        elif self.config.backend == BrowserBackend.SELENIUM:
            self.page.back()

    async def go_forward(self):
        """Navigate forward."""
        if self.config.backend == BrowserBackend.PLAYWRIGHT:
            await self.page.go_forward()
        elif self.config.backend == BrowserBackend.SELENIUM:
            self.page.forward()

    async def reload(self):
        """Reload current page."""
        if self.config.backend == BrowserBackend.PLAYWRIGHT:
            await self.page.reload()
        elif self.config.backend == BrowserBackend.SELENIUM:
            self.page.refresh()

    # === Element Interaction ===

    async def click(self, selector: str, timeout: int = None) -> bool:
        """Click an element."""
        timeout = timeout or self.config.timeout

        try:
            if self.config.backend == BrowserBackend.PLAYWRIGHT:
                await self.page.click(selector, timeout=timeout)
            elif self.config.backend == BrowserBackend.SELENIUM:
                from selenium.webdriver.common.by import By
                element = self.page.find_element(By.CSS_SELECTOR, selector)
                element.click()
            return True
        except Exception as e:
            return False

    async def type_text(self, selector: str, text: str, delay: int = 50) -> bool:
        """Type text into an element."""
        try:
            if self.config.backend == BrowserBackend.PLAYWRIGHT:
                await self.page.fill(selector, text)
            elif self.config.backend == BrowserBackend.SELENIUM:
                from selenium.webdriver.common.by import By
                from selenium.webdriver.common.keys import Keys
                element = self.page.find_element(By.CSS_SELECTOR, selector)
                element.clear()
                element.send_keys(text)
            return True
        except Exception:
            return False

    async def press_key(self, key: str) -> bool:
        """Press a keyboard key."""
        try:
            if self.config.backend == BrowserBackend.PLAYWRIGHT:
                await self.page.keyboard.press(key)
            elif self.config.backend == BrowserBackend.SELENIUM:
                from selenium.webdriver.common.keys import Keys
                key_map = {
                    "Enter": Keys.RETURN,
                    "Tab": Keys.TAB,
                    "Escape": Keys.ESCAPE,
                    "Backspace": Keys.BACKSPACE,
                }
                from selenium.webdriver.common.action_chains import ActionChains
                ActionChains(self.page).send_keys(key_map.get(key, key)).perform()
            return True
        except Exception:
            return False

    async def select_option(self, selector: str, value: str) -> bool:
        """Select an option from dropdown."""
        try:
            if self.config.backend == BrowserBackend.PLAYWRIGHT:
                await self.page.select_option(selector, value)
            elif self.config.backend == BrowserBackend.SELENIUM:
                from selenium.webdriver.common.by import By
                from selenium.webdriver.support.ui import Select
                element = self.page.find_element(By.CSS_SELECTOR, selector)
                Select(element).select_by_value(value)
            return True
        except Exception:
            return False

    async def scroll(self, x: int = 0, y: int = 500) -> bool:
        """Scroll the page."""
        try:
            if self.config.backend == BrowserBackend.PLAYWRIGHT:
                await self.page.mouse.wheel(x, y)
            elif self.config.backend == BrowserBackend.SELENIUM:
                self.page.execute_script(f"window.scrollBy({x}, {y})")
            return True
        except Exception:
            return False

    async def hover(self, selector: str) -> bool:
        """Hover over an element."""
        try:
            if self.config.backend == BrowserBackend.PLAYWRIGHT:
                await self.page.hover(selector)
            elif self.config.backend == BrowserBackend.SELENIUM:
                from selenium.webdriver.common.by import By
                from selenium.webdriver.common.action_chains import ActionChains
                element = self.page.find_element(By.CSS_SELECTOR, selector)
                ActionChains(self.page).move_to_element(element).perform()
            return True
        except Exception:
            return False

    # === Content Extraction ===

    async def get_text(self, selector: str = None) -> str:
        """Get text content."""
        try:
            if self.config.backend == BrowserBackend.PLAYWRIGHT:
                if selector:
                    element = await self.page.query_selector(selector)
                    return await element.text_content() if element else ""
                return await self.page.inner_text("body")
            elif self.config.backend == BrowserBackend.SELENIUM:
                from selenium.webdriver.common.by import By
                if selector:
                    element = self.page.find_element(By.CSS_SELECTOR, selector)
                    return element.text
                return self.page.find_element(By.TAG_NAME, "body").text
        except Exception:
            return ""

    async def get_attribute(self, selector: str, attribute: str) -> str:
        """Get attribute value of an element."""
        try:
            if self.config.backend == BrowserBackend.PLAYWRIGHT:
                return await self.page.get_attribute(selector, attribute)
            elif self.config.backend == BrowserBackend.SELENIUM:
                from selenium.webdriver.common.by import By
                element = self.page.find_element(By.CSS_SELECTOR, selector)
                return element.get_attribute(attribute)
        except Exception:
            return ""

    async def get_value(self, selector: str) -> str:
        """Get input value."""
        return await self.get_attribute(selector, "value")

    async def get_html(self, selector: str = "body") -> str:
        """Get HTML content."""
        try:
            if self.config.backend == BrowserBackend.PLAYWRIGHT:
                return await self.page.inner_html(selector)
            elif self.config.backend == BrowserBackend.SELENIUM:
                from selenium.webdriver.common.by import By
                element = self.page.find_element(By.CSS_SELECTOR, selector)
                return element.get_attribute("innerHTML")
        except Exception:
            return ""

    async def _find_elements(self, selector: str = "a, button, input, textarea, select") -> list[ElementInfo]:
        """Find all matching elements."""
        elements = []

        try:
            if self.config.backend == BrowserBackend.PLAYWRIGHT:
                locators = await self.page.query_selector_all(selector)
                for i, locator in enumerate(locators):
                    tag = await locator.evaluate("el => el.tagName.toLowerCase()")
                    text = await locator.text_content() or ""
                    visible = await locator.is_visible()
                    elements.append(ElementInfo(
                        tag=tag,
                        text=text.strip()[:100],
                        attributes={},
                        selector=f"{tag}:nth-child({i+1})",
                        index=i,
                        visible=visible,
                        enabled=True
                    ))

            elif self.config.backend == BrowserBackend.SELENIUM:
                from selenium.webdriver.common.by import By
                found = self.page.find_elements(By.CSS_SELECTOR, selector)
                for i, elem in enumerate(found):
                    elements.append(ElementInfo(
                        tag=elem.tag_name,
                        text=elem.text[:100] if elem.text else "",
                        attributes={},
                        selector=f"{elem.tag_name}:nth-child({i+1})",
                        index=i,
                        visible=elem.is_displayed(),
                        enabled=elem.is_enabled()
                    ))
        except Exception:
            pass

        return elements

    # === Scraping ===

    async def scrape(self, url: str, selectors: dict = None) -> ScrapedContent:
        """Scrape content from a URL."""
        result = await self.navigate(url)

        if not result.success:
            return ScrapedContent(
                url=url,
                title="",
                text="",
                html="",
                links=[],
                images=[],
                metadata={"error": result.error}
            )

        text = await self.get_text()
        html = await self.get_html()
        title = result.title

        # Extract links
        links = await self._extract_links()

        # Extract images
        images = await self._extract_images()

        # Extract metadata
        metadata = await self._extract_metadata()

        # Extract custom selectors
        custom_data = {}
        if selectors:
            for key, selector in selectors.items():
                custom_data[key] = await self.get_text(selector)

        return ScrapedContent(
            url=url,
            title=title,
            text=text,
            html=html,
            links=links,
            images=images,
            metadata={**metadata, **custom_data}
        )

    async def _extract_links(self) -> list[dict]:
        """Extract all links from page."""
        links = []

        try:
            if self.config.backend == BrowserBackend.PLAYWRIGHT:
                link_elements = await self.page.query_selector_all("a[href]")
                for link in link_elements:
                    href = await link.get_attribute("href")
                    text = await link.text_content()
                    if href:
                        links.append({
                            "url": href,
                            "text": (text or "").strip()[:100]
                        })

            elif self.config.backend == BrowserBackend.SELENIUM:
                from selenium.webdriver.common.by import By
                link_elements = self.page.find_elements(By.TAG_NAME, "a")
                for link in link_elements:
                    href = link.get_attribute("href")
                    if href:
                        links.append({
                            "url": href,
                            "text": link.text[:100] if link.text else ""
                        })
        except Exception:
            pass

        return links

    async def _extract_images(self) -> list[dict]:
        """Extract all images from page."""
        images = []

        try:
            if self.config.backend == BrowserBackend.PLAYWRIGHT:
                img_elements = await self.page.query_selector_all("img")
                for img in img_elements:
                    src = await img.get_attribute("src")
                    alt = await img.get_attribute("alt")
                    if src:
                        images.append({
                            "url": src,
                            "alt": alt or ""
                        })

            elif self.config.backend == BrowserBackend.SELENIUM:
                from selenium.webdriver.common.by import By
                img_elements = self.page.find_elements(By.TAG_NAME, "img")
                for img in img_elements:
                    src = img.get_attribute("src")
                    if src:
                        images.append({
                            "url": src,
                            "alt": img.get_attribute("alt") or ""
                        })
        except Exception:
            pass

        return images

    async def _extract_metadata(self) -> dict:
        """Extract page metadata."""
        metadata = {}

        try:
            if self.config.backend == BrowserBackend.PLAYWRIGHT:
                # Get meta tags
                meta_tags = await self.page.query_selector_all("meta")
                for meta in meta_tags:
                    name = await meta.get_attribute("name")
                    property_attr = await meta.get_attribute("property")
                    content = await meta.get_attribute("content")
                    if content:
                        key = name or property_attr
                        if key:
                            metadata[key] = content

            elif self.config.backend == BrowserBackend.SELENIUM:
                from selenium.webdriver.common.by import By
                meta_elements = self.page.find_elements(By.TAG_NAME, "meta")
                for meta in meta_elements:
                    name = meta.get_attribute("name")
                    property_attr = meta.get_attribute("property")
                    content = meta.get_attribute("content")
                    if content:
                        key = name or property_attr
                        if key:
                            metadata[key] = content
        except Exception:
            pass

        return metadata

    # === Forms ===

    async def fill_form(self, form_data: dict) -> bool:
        """
        Fill a form with data.

        form_data: {selector: value}
        """
        success = True
        for selector, value in form_data.items():
            if not await self.type_text(selector, str(value)):
                success = False
        return success

    async def submit_form(self, selector: str = "form") -> bool:
        """Submit a form."""
        try:
            if self.config.backend == BrowserBackend.PLAYWRIGHT:
                await self.page.evaluate(f"document.querySelector('{selector}').submit()")
            elif self.config.backend == BrowserBackend.SELENIUM:
                from selenium.webdriver.common.by import By
                form = self.page.find_element(By.CSS_SELECTOR, selector)
                form.submit()
            return True
        except Exception:
            return False

    # === Screenshots ===

    async def screenshot(self, path: str = None, full_page: bool = False) -> str:
        """Take a screenshot."""
        if not path:
            os.makedirs(self.config.screenshots_dir, exist_ok=True)
            timestamp = int(time.time() * 1000)
            path = os.path.join(self.config.screenshots_dir, f"screenshot_{timestamp}.png")

        try:
            if self.config.backend == BrowserBackend.PLAYWRIGHT:
                await self.page.screenshot(path=path, full_page=full_page)
            elif self.config.backend == BrowserBackend.SELENIUM:
                self.page.save_screenshot(path)
            return path
        except Exception:
            return ""

    # === JavaScript Execution ===

    async def execute_js(self, script: str) -> Any:
        """Execute JavaScript in the page."""
        try:
            if self.config.backend == BrowserBackend.PLAYWRIGHT:
                return await self.page.evaluate(script)
            elif self.config.backend == BrowserBackend.SELENIUM:
                return self.page.execute_script(script)
        except Exception as e:
            return {"error": str(e)}

    # === Cookies & Storage ===

    async def get_cookies(self) -> list[dict]:
        """Get all cookies."""
        try:
            if self.config.backend == BrowserBackend.PLAYWRIGHT:
                return await self.context.cookies()
            elif self.config.backend == BrowserBackend.SELENIUM:
                return self.page.get_cookies()
        except Exception:
            return []

    async def set_cookie(self, name: str, value: str, domain: str = None):
        """Set a cookie."""
        try:
            if self.config.backend == BrowserBackend.PLAYWRIGHT:
                cookie = {"name": name, "value": value}
                if domain:
                    cookie["domain"] = domain
                await self.context.add_cookies([cookie])
            elif self.config.backend == BrowserBackend.SELENIUM:
                cookie = {"name": name, "value": value}
                if domain:
                    cookie["domain"] = domain
                self.page.add_cookie(cookie)
        except Exception:
            pass

    async def clear_cookies(self):
        """Clear all cookies."""
        try:
            if self.config.backend == BrowserBackend.PLAYWRIGHT:
                await self.context.clear_cookies()
            elif self.config.backend == BrowserBackend.SELENIUM:
                self.page.delete_all_cookies()
        except Exception:
            pass

    # === Utility ===

    async def wait_for_selector(self, selector: str, timeout: int = None) -> bool:
        """Wait for an element to appear."""
        timeout = timeout or self.config.timeout

        try:
            if self.config.backend == BrowserBackend.PLAYWRIGHT:
                await self.page.wait_for_selector(selector, timeout=timeout)
            elif self.config.backend == BrowserBackend.SELENIUM:
                from selenium.webdriver.common.by import By
                from selenium.webdriver.support.ui import WebDriverWait
                from selenium.webdriver.support import expected_conditions as EC
                WebDriverWait(self.page, timeout / 1000).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, selector))
                )
            return True
        except Exception:
            return False

    async def wait_for_load(self, state: str = "load"):
        """Wait for page load state."""
        if self.config.backend == BrowserBackend.PLAYWRIGHT:
            await self.page.wait_for_load_state(state)

    def get_history(self) -> list[dict]:
        """Get navigation history."""
        return self._history

    async def get_current_url(self) -> str:
        """Get current URL."""
        try:
            if self.config.backend == BrowserBackend.PLAYWRIGHT:
                return self.page.url
            elif self.config.backend == BrowserBackend.SELENIUM:
                return self.page.current_url
        except Exception:
            return ""


class WebScraper:
    """
    High-level web scraper with common patterns.
    """

    def __init__(self, config: BrowserConfig = None):
        self.browser = BrowserAutomation(config)
        self._cache: dict[str, ScrapedContent] = {}

    async def scrape_url(self, url: str, use_cache: bool = True) -> ScrapedContent:
        """Scrape a single URL."""
        if use_cache and url in self._cache:
            return self._cache[url]

        content = await self.browser.scrape(url)
        self._cache[url] = content
        return content

    async def scrape_multiple(self, urls: list[str], delay: float = 1.0) -> list[ScrapedContent]:
        """Scrape multiple URLs with delay."""
        results = []
        for url in urls:
            result = await self.scrape_url(url)
            results.append(result)
            if delay > 0:
                await asyncio.sleep(delay)
        return results

    async def search_and_scrape(self, query: str, num_results: int = 5) -> list[ScrapedContent]:
        """Search and scrape results."""
        # Google search
        search_url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
        await self.browser.navigate(search_url)

        # Extract result links
        links = await self.browser._extract_links()
        result_links = [
            link["url"] for link in links
            if link["url"].startswith("http") and "google" not in link["url"]
        ][:num_results]

        # Scrape each result
        return await self.browser.scrape_multiple(result_links)

    async def extract_article(self, url: str) -> dict:
        """Extract article content from URL."""
        content = await self.scrape_url(url)

        # Try common article selectors
        article_selectors = [
            "article",
            ".article-content",
            ".post-content",
            ".entry-content",
            "#article",
            "main",
        ]

        article_text = ""
        for selector in article_selectors:
            text = await self.browser.get_text(selector)
            if len(text) > len(article_text):
                article_text = text

        return {
            "url": url,
            "title": content.title,
            "text": article_text or content.text,
            "metadata": content.metadata
        }


class FormFiller:
    """
    Automated form filling utility.
    """

    def __init__(self, browser: BrowserAutomation):
        self.browser = browser

    async def fill_contact_form(self, data: dict) -> bool:
        """Fill a contact form."""
        selectors = {
            "name": ["#name", "[name='name']", "[name='姓名']"],
            "email": ["#email", "[name='email']", "[type='email']"],
            "phone": ["#phone", "[name='phone']", "[type='tel']"],
            "message": ["#message", "[name='message']", "textarea"],
        }

        success = True
        for field, value in data.items():
            for selector in selectors.get(field, []):
                if await self.browser.type_text(selector, value):
                    break
            else:
                success = False

        return success

    async def fill_login_form(self, username: str, password: str,
                             username_selector: str = "#username",
                             password_selector: str = "#password") -> bool:
        """Fill a login form."""
        if not await self.browser.type_text(username_selector, username):
            return False
        if not await self.browser.type_text(password_selector, password):
            return False
        return True


# === Convenience Functions ===

async def quick_scrape(url: str) -> dict:
    """Quick scrape of a URL."""
    browser = BrowserAutomation()
    try:
        await browser.start()
        content = await browser.scrape(url)
        return {
            "url": content.url,
            "title": content.title,
            "text": content.text[:5000],
            "links_count": len(content.links),
            "images_count": len(content.images)
        }
    finally:
        await browser.stop()


async def take_screenshot(url: str, path: str = None) -> str:
    """Take a screenshot of a URL."""
    browser = BrowserAutomation()
    try:
        await browser.start()
        await browser.navigate(url)
        return await browser.screenshot(path)
    finally:
        await browser.stop()


async def fill_and_submit(url: str, form_data: dict, submit_selector: str = "form") -> bool:
    """Fill and submit a form."""
    browser = BrowserAutomation()
    try:
        await browser.start()
        await browser.navigate(url)
        await browser.fill_form(form_data)
        return await browser.submit_form(submit_selector)
    finally:
        await browser.stop()


def main(argv=None):
    import asyncio

    def _add_browser_args(p):
        p.add_argument(
            "--backend",
            default="playwright",
            choices=[b.value for b in BrowserBackend],
            help="Browser automation backend",
        )
        p.add_argument("--browser-type", default="chromium", help="Browser type (chromium, firefox, webkit)")
        p.add_argument("--no-headless", action="store_true", help="Run in headed mode")
        p.add_argument("--viewport-width", type=int, default=1280, help="Viewport width")
        p.add_argument("--viewport-height", type=int, default=720, help="Viewport height")
        p.add_argument("--timeout", type=int, default=30000, help="Timeout in milliseconds")
        p.add_argument("--proxy", default=None, help="Proxy server URL")
        p.add_argument("--user-agent", default=None, help="Custom user agent")

    parser = argparse.ArgumentParser(description="Browser automation CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("config", help="Show browser configuration (no browser started)")
    _add_browser_args(p)

    p = sub.add_parser("scrape", help="Scrape content from a URL")
    p.add_argument("--url", required=True, help="URL to scrape")
    _add_browser_args(p)

    p = sub.add_parser("screenshot", help="Capture a screenshot of a URL")
    p.add_argument("--url", required=True, help="URL to capture")
    p.add_argument("--path", default=None, help="Output screenshot path")
    _add_browser_args(p)

    args = parser.parse_args(argv)

    try:
        config = BrowserConfig(
            backend=BrowserBackend(args.backend),
            headless=not args.no_headless,
            browser_type=args.browser_type,
            viewport_width=args.viewport_width,
            viewport_height=args.viewport_height,
            timeout=args.timeout,
            proxy=args.proxy,
            user_agent=args.user_agent,
        )

        if args.command == "config":
            print(json.dumps({
                "backend": config.backend.value,
                "headless": config.headless,
                "browser_type": config.browser_type,
                "viewport_width": config.viewport_width,
                "viewport_height": config.viewport_height,
                "timeout": config.timeout,
                "proxy": config.proxy,
                "user_agent": config.user_agent,
            }, indent=2))
            return 0
        elif args.command == "scrape":
            async def _scrape():
                ws = WebScraper(config)
                content = await ws.scrape_url(args.url)
                return {
                    "url": content.url,
                    "title": content.title,
                    "text_length": len(content.text),
                    "html_length": len(content.html),
                    "links_count": len(content.links),
                    "images_count": len(content.images),
                    "metadata": content.metadata,
                }
            result = asyncio.run(_scrape())
            print(json.dumps(result, indent=2, default=str))
            return 0
        elif args.command == "screenshot":
            async def _shot():
                browser = BrowserAutomation(config)
                await browser.start()
                try:
                    await browser.navigate(args.url)
                    return await browser.screenshot(args.path)
                finally:
                    await browser.stop()
            path = asyncio.run(_shot())
            print(json.dumps({"screenshot": path}, indent=2))
            return 0 if path else 1
        else:
            parser.error("Unknown command")
            return 2
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    main()


# --- Console tracing, auto-debug, and dead-code extraction submodules ---
from .tracer import (
    ConsoleTracer,
    TraceEntry,
    TraceLevel,
)
from .auto_debug import (
    AutoDebugger,
    DeadCodeDetector,
    IssueCategory,
    FixAction,
    trace_and_debug,
    extract_deadcode,
)

__all__ = [
    # ...existing public names if any...
    "ConsoleTracer",
    "TraceEntry",
    "TraceLevel",
    "AutoDebugger",
    "DeadCodeDetector",
    "IssueCategory",
    "FixAction",
    "trace_and_debug",
    "extract_deadcode",
]
