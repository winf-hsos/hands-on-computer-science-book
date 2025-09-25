-- filters/gloss.lua
-- Macht \gloss{Begriff}{Definition}{Kategorie?} im Text sichtbar
-- und verlinkt auf glossary.html#gl-<slug(Begriff)> (nur HTML).

local function slugify(s)
  s = s:lower()
  s = s:gsub("%s+", "-")
  s = s:gsub("[^%w%-%_]", "")
  return s
end

function RawInline(el)
  if el.format ~= "tex" then return nil end
  -- akzeptiert 2 oder 3 Argumente (Kategorie optional)
  local term, def, cat =
    el.text:match("\\gloss%{(.-)%}%{(.-)%}%{(.-)%}")  -- 3-arg
  if not term then
    term, def = el.text:match("\\gloss%{(.-)%}%{(.-)%}") -- 2-arg
  end
  if not term then return nil end

  local term_span = pandoc.Span(pandoc.Str(term), { class = "gloss-term" })

  if FORMAT:match("html") then
    local target = "glossary.html#gl-" .. slugify(term)
    return pandoc.Link(term_span, target, term)
  else
    -- in PDF/EPUB einfach nur den Begriff ausgeben
    return term_span
  end
end
