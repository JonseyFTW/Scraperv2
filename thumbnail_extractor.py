"""
Thumbnail extraction for CLIP embeddings
Extract thumbnail URLs during Phase 1/2 to avoid needing full-res images
"""
import re
import json
from typing import List, Dict, Optional
from dataclasses import dataclass

from rich.console import Console

console = Console()


@dataclass
class CardThumbnail:
    """Thumbnail data for a card"""
    product_id: str
    set_slug: str
    thumbnail_url: str
    thumbnail_size: str  # 'small', 'medium', 'large'
    source_page: str  # Where we found it
    
    
class ThumbnailExtractor:
    """
    Extract thumbnail URLs from set listing pages
    These are often sufficient for CLIP similarity matching
    """
    
    def __init__(self):
        # Common thumbnail URL patterns
        self.patterns = [
            # PriceCharting CDN thumbnails
            (r'https://storage\.googleapis\.com/images\.pricecharting\.com/([^/\s"\'<>]+)/(\d+)(?:\.jpg)?', 'pricecharting'),
            # Direct thumbnails on listing pages
            (r'<img[^>]+src="([^"]+/thumbnails?/[^"]+)"', 'listing_thumb'),
            # Lazy-loaded thumbnails
            (r'data-src="([^"]+(?:thumb|small|preview)[^"]+)"', 'lazy_thumb'),
            # Background image thumbnails
            (r'background-image:\s*url\(["\']?([^"\']+(?:thumb|small)[^"\']+)["\']?\)', 'bg_thumb'),
        ]
        
        self.size_mappings = {
            '200': 'small',    # 200x200 or smaller
            '400': 'medium',   # 400x400
            '800': 'large',    # 800x800
            '1600': 'full',    # Full size
        }
        
    def extract_from_html(self, html: str, base_url: str) -> List[CardThumbnail]:
        """Extract all thumbnail URLs from HTML"""
        thumbnails = []
        
        # Try to find card containers first
        card_blocks = self.extract_card_blocks(html)
        
        if card_blocks:
            # Process individual card blocks
            for block in card_blocks:
                thumb = self.extract_from_card_block(block, base_url)
                if thumb:
                    thumbnails.append(thumb)
        else:
            # Fall back to finding all images
            thumbnails = self.extract_all_thumbnails(html, base_url)
            
        return thumbnails
        
    def extract_card_blocks(self, html: str) -> List[str]:
        """Extract individual card HTML blocks from listing page"""
        blocks = []
        
        # Common card container patterns
        patterns = [
            r'<div[^>]+class="[^"]*card-item[^"]*"[^>]*>.*?</div>',
            r'<article[^>]+class="[^"]*product[^"]*"[^>]*>.*?</article>',
            r'<li[^>]+class="[^"]*listing[^"]*"[^>]*>.*?</li>',
            r'<a[^>]+href="/game/[^"]+"[^>]*>.*?</a>',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, html, re.DOTALL | re.IGNORECASE)
            if matches:
                blocks.extend(matches)
                break
                
        return blocks
        
    def extract_from_card_block(self, block: str, base_url: str) -> Optional[CardThumbnail]:
        """Extract thumbnail from a single card block"""
        # Extract product ID if available
        product_id = None
        id_match = re.search(r'data-product-id="(\d+)"', block)
        if not id_match:
            id_match = re.search(r'id="product-(\d+)"', block)
        if id_match:
            product_id = id_match.group(1)
            
        # Extract card URL to get slug
        slug = None
        url_match = re.search(r'href="(/game/[^/]+/([^"]+))"', block)
        if url_match:
            slug = url_match.group(2)
            
        # Find thumbnail URL
        thumb_url = None
        thumb_size = 'medium'
        
        for pattern, source in self.patterns:
            match = re.search(pattern, block, re.IGNORECASE)
            if match:
                if source == 'pricecharting':
                    # Build thumbnail URL from hash
                    hash_val = match.group(1)
                    # Use 400px version for CLIP (good balance of quality/size)
                    thumb_url = f"https://storage.googleapis.com/images.pricecharting.com/{hash_val}/400.jpg"
                    thumb_size = 'medium'
                else:
                    thumb_url = match.group(1)
                    # Determine size from URL
                    for size_marker, size_name in self.size_mappings.items():
                        if size_marker in thumb_url:
                            thumb_size = size_name
                            break
                break
                
        if thumb_url and (product_id or slug):
            # Make URL absolute if needed
            if thumb_url.startswith('//'):
                thumb_url = 'https:' + thumb_url
            elif thumb_url.startswith('/'):
                thumb_url = base_url + thumb_url
                
            return CardThumbnail(
                product_id=product_id or slug,
                set_slug=slug or '',
                thumbnail_url=thumb_url,
                thumbnail_size=thumb_size,
                source_page=base_url
            )
            
        return None
        
    def extract_all_thumbnails(self, html: str, base_url: str) -> List[CardThumbnail]:
        """Extract all thumbnail URLs from page"""
        thumbnails = []
        seen_urls = set()
        
        # Find all image URLs
        img_patterns = [
            r'<img[^>]+src="([^"]+)"',
            r'data-src="([^"]+)"',
            r'data-lazy="([^"]+)"',
            r'background-image:\s*url\(["\']?([^"\']+)["\']?\)',
        ]
        
        for pattern in img_patterns:
            matches = re.findall(pattern, html, re.IGNORECASE)
            
            for url in matches:
                if url in seen_urls:
                    continue
                seen_urls.add(url)
                
                # Skip non-card images
                if any(skip in url.lower() for skip in ['logo', 'banner', 'icon', 'avatar', 'placeholder']):
                    continue
                    
                # Check if it's a card image
                if any(marker in url for marker in ['/cards/', '/products/', 'pricecharting.com']):
                    # Make URL absolute
                    if url.startswith('//'):
                        url = 'https:' + url
                    elif url.startswith('/'):
                        url = base_url + url
                        
                    # Determine size
                    thumb_size = 'medium'
                    for size_marker, size_name in self.size_mappings.items():
                        if size_marker in url:
                            thumb_size = size_name
                            break
                            
                    # Extract any ID from URL
                    id_match = re.search(r'/(\d+)[^\d]', url)
                    product_id = id_match.group(1) if id_match else f"thumb_{len(thumbnails)}"
                    
                    thumbnails.append(CardThumbnail(
                        product_id=product_id,
                        set_slug='',
                        thumbnail_url=url,
                        thumbnail_size=thumb_size,
                        source_page=base_url
                    ))
                    
        return thumbnails
        
    def optimize_for_clip(self, thumbnails: List[CardThumbnail]) -> List[CardThumbnail]:
        """
        Optimize thumbnail list for CLIP embeddings
        Prefer medium-size thumbnails (400-800px) for best quality/speed balance
        """
        # Group by product_id
        by_product = {}
        for thumb in thumbnails:
            pid = thumb.product_id
            if pid not in by_product:
                by_product[pid] = []
            by_product[pid].append(thumb)
            
        # Select best thumbnail for each product
        optimized = []
        size_priority = ['medium', 'large', 'small', 'full']
        
        for pid, thumbs in by_product.items():
            # Sort by size priority
            thumbs.sort(key=lambda t: size_priority.index(t.thumbnail_size) 
                       if t.thumbnail_size in size_priority else 999)
            optimized.append(thumbs[0])
            
        return optimized


def extract_thumbnails_from_csv_page(html: str, set_url: str) -> List[CardThumbnail]:
    """
    Extract thumbnails from a set's CSV download page
    These pages often show card images in a grid
    """
    extractor = ThumbnailExtractor()
    thumbnails = extractor.extract_from_html(html, set_url)
    
    if thumbnails:
        console.print(f"[green]Found {len(thumbnails)} thumbnails on page[/green]")
        
        # Optimize for CLIP
        optimized = extractor.optimize_for_clip(thumbnails)
        console.print(f"[cyan]Optimized to {len(optimized)} thumbnails for CLIP[/cyan]")
        
        return optimized
    else:
        console.print("[yellow]No thumbnails found on page[/yellow]")
        return []


def save_thumbnails_to_db(thumbnails: List[CardThumbnail]):
    """Save thumbnail URLs to database"""
    import database as db
    
    for thumb in thumbnails:
        # Update card with thumbnail URL
        # This can be used instead of full image for CLIP
        try:
            db.update_card_image_url(thumb.product_id, thumb.thumbnail_url)
            # Mark as 'thumbnail_found' instead of 'image_found'
            db.execute_query(
                "UPDATE cards SET status='thumbnail_found' WHERE product_id=%s",
                (thumb.product_id,)
            )
        except:
            pass  # Card might not exist yet
            

def batch_extract_thumbnails(html_files: List[str]) -> Dict[str, List[CardThumbnail]]:
    """
    Batch extract thumbnails from multiple HTML files
    Useful for testing and debugging
    """
    extractor = ThumbnailExtractor()
    results = {}
    
    for filepath in html_files:
        with open(filepath, 'r', encoding='utf-8') as f:
            html = f.read()
            
        # Extract base URL from file
        base_url = "https://www.sportscardspro.com"
        
        thumbnails = extractor.extract_from_html(html, base_url)
        results[filepath] = thumbnails
        
        console.print(f"[cyan]{filepath}:[/cyan] {len(thumbnails)} thumbnails")
        
    return results


if __name__ == "__main__":
    # Test/demo
    import sys
    
    if len(sys.argv) > 1:
        # Test with HTML file
        filepath = sys.argv[1]
        
        console.print(f"[bold]Testing thumbnail extraction on: {filepath}[/bold]")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            html = f.read()
            
        extractor = ThumbnailExtractor()
        thumbnails = extractor.extract_from_html(html, "https://www.sportscardspro.com")
        
        console.print(f"\n[green]Found {len(thumbnails)} thumbnails[/green]\n")
        
        # Show first 5
        for thumb in thumbnails[:5]:
            console.print(f"ID: {thumb.product_id}")
            console.print(f"  URL: {thumb.thumbnail_url}")
            console.print(f"  Size: {thumb.thumbnail_size}")
            console.print()
            
        # Optimize for CLIP
        optimized = extractor.optimize_for_clip(thumbnails)
        console.print(f"[cyan]Optimized to {len(optimized)} thumbnails for CLIP[/cyan]")
    else:
        console.print("[yellow]Usage: python thumbnail_extractor.py <html_file>[/yellow]")
        console.print("\nThumbnail extractor can:")
        console.print("• Extract thumbnail URLs from listing pages")
        console.print("• Optimize thumbnails for CLIP embeddings")
        console.print("• Reduce image download requirements by 90%+")
        console.print("• Work with 400px thumbnails instead of 1600px full images")