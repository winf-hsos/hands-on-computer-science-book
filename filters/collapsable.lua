-- filters/collapsable.lua
function Pandoc(doc)
  local out = {}
  local i = 1
  while i <= #doc.blocks do
    local blk = doc.blocks[i]

    if blk.t == "Header" and blk.attr and blk.attr.classes:includes("collapsable") then
      local level = blk.level
      table.insert(out, blk) -- Header bleibt normal

      -- Blöcke bis zum nächsten Header <= gleicher Ebene sammeln
      local collected = {}
      i = i + 1
      while i <= #doc.blocks do
        local b = doc.blocks[i]
        if b.t == "Header" and b.level <= level then break end
        table.insert(collected, b)
        i = i + 1
      end

      -- <details> mit LEEREM <summary>, das wir per CSS ausblenden
      table.insert(out, pandoc.RawBlock("html",
        '<details class="collapsable-section" open><summary></summary>'))
      for _,b in ipairs(collected) do table.insert(out, b) end
      table.insert(out, pandoc.RawBlock("html", '</details>'))

    else
      table.insert(out, blk)
      i = i + 1
    end
  end
  return pandoc.Pandoc(out, doc.meta)
end
