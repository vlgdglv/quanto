# data/rest_public.py
import requests
from typing import List, Dict, Any

class OKXPublicREST:
    def __init__(self, base_url: str = "https://www.okx.com", timeout: int = 10):
        self.base = base_url.rstrip("/")
        self.sess = requests.Session()
        self.timeout = timeout

    def candles(self, instId: str, bar: str = "1m", limit: int = 100) -> List[List[str]]:
        """
        GET /api/v5/market/candles
        Returns list of [ts, o, h, l, c, vol, volCcy, volCcyQuote, confirm, ...]
        """
        url = f"{self.base}/api/v5/market/candles"
        params = {"instId": instId, "bar": bar, "limit": str(limit)}
        r = self.sess.get(url, params=params, timeout=self.timeout)
        r.raise_for_status()
        return r.json().get("data", [])

    def trades(self, instId: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        GET /api/v5/market/trades
        """
        url = f"{self.base}/api/v5/market/trades"
        params = {"instId": instId, "limit": str(limit)}
        r = self.sess.get(url, params=params, timeout=self.timeout)
        r.raise_for_status()
        return r.json().get("data", [])

    def books(self, instId: str, sz: int = 5) -> Dict[str, Any]:
        """
        GET /api/v5/market/books?sz=5
        """
        url = f"{self.base}/api/v5/market/books"
        params = {"instId": instId, "sz": str(sz)}
        r = self.sess.get(url, params=params, timeout=self.timeout)
        r.raise_for_status()
        return r.json().get("data", [])[0] if r.json().get("data") else {}
