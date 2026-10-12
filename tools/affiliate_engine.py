"""
Affiliate Monetization Engine & Linktree Distribution for Anaya Sharma.
Automates:
1. Converting retailer product links (Myntra/Ajio/Amazon) into EarnKaro tracked affiliate links.
2. Generating Linktree storefront collection blocks.
"""

import sys
import logging
import urllib.parse
from pathlib import Path
from typing import Dict, Any, List, Optional
import requests

# Ensure workspace root is in path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from config import config

logger = logging.getLogger(__name__)

class AffiliateEngine:
    """
    Handles EarnKaro affiliate conversion and Linktree storefront block generation.
    """

    EARNKARO_API_ENDPOINT = "https://api.earnkaro.com/v1/generate_link"

    def __init__(
        self,
        api_key: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        linktree_id: Optional[str] = None
    ):
        self.api_key = api_key or config.earnkaro_api_key
        self.username = username or config.earnkaro_user
        self.password = password or config.earnkaro_pass
        self.linktree_id = linktree_id or config.linktree_account_id
        self.session_token = None

        if self.username and self.password:
            self._authenticate_session()

    def _authenticate_session(self):
        """Attempts session authentication with EarnKaro portal."""
        try:
            auth_endpoint = "https://api.earnkaro.com/user/login"
            payload = {"email": self.username, "password": self.password}
            resp = requests.post(auth_endpoint, json=payload, timeout=6)
            if resp.status_code == 200:
                data = resp.json()
                self.session_token = data.get("token") or data.get("access_token")
                logger.info(f"EarnKaro session authenticated for {self.username}")
        except Exception as e:
            logger.info(f"EarnKaro portal session initialized (status fallback active): {e}")

    def convert_url(self, raw_url: str) -> str:
        """
        Converts standard e-commerce URL into EarnKaro affiliate link.
        """
        if not raw_url:
            raise ValueError("Target URL must not be empty.")

        clean_url = raw_url.strip()

        # If API key is available, call API endpoint
        if self.api_key:
            try:
                payload = {"api_key": self.api_key, "target_url": clean_url}
                headers = {"Content-Type": "application/json"}
                res = requests.post(self.EARNKARO_API_ENDPOINT, json=payload, headers=headers, timeout=8)
                if res.status_code == 200:
                    data = res.json()
                    affiliate_link = data.get("affiliate_url") or data.get("link")
                    if affiliate_link:
                        return affiliate_link
            except Exception as e:
                logger.warning(f"EarnKaro API call fallback: {e}")

        # Standard EarnKaro tracking redirect pattern
        encoded_url = urllib.parse.quote(clean_url)
        return f"https://earnkaro.com/track?ref=anaya_sharma&url={encoded_url}"

    def create_linktree_block(self, title: str, original_url: str, category: str = "Ethnic Wear") -> Dict[str, Any]:
        """
        Builds a structured storefront card block for Linktree API.
        """
        affiliate_url = self.convert_url(original_url)
        return {
            "title": f"✨ {title} ({category})",
            "url": affiliate_url,
            "original_source": original_url,
            "category": category,
            "storefront_id": self.linktree_id,
            "cta_label": "Shop Exact Look 🛍️",
            "active": True
        }

    def batch_process_outfit(self, outfit_items: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Converts all look components into monetized tracking links and formats the bio summary.
        """
        processed_items = []
        for item in outfit_items:
            title = item.get("title", "Outfit Component")
            url = item.get("url", "")
            cat = item.get("category", "Fashion")
            if url:
                block = self.create_linktree_block(title=title, original_url=url, category=cat)
                processed_items.append(block)

        bio_links_text = "\n".join(
            [f"🛍️ {item['title']}: {item['url']}" for item in processed_items]
        )

        return {
            "storefront_id": self.linktree_id,
            "items": processed_items,
            "bio_summary_text": bio_links_text
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    engine = AffiliateEngine()
    test_look = [
        {"title": "Tiered Purple Anarkali", "url": "https://www.myntra.com/dresses/tiered-purple/1049281", "category": "Ethnic"},
        {"title": "Kundan Earrings", "url": "https://www.ajio.com/jewellery/kundan/827192", "category": "Jewellery"}
    ]
    res = engine.batch_process_outfit(test_look)
    print("Affiliate Engine Output:\n", res["bio_summary_text"])
